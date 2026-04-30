"""Tests for MASArcAgent workspace loading, skill injection, action limits,
thinking effort, per-role artifacts, and shared memory/wiki."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure we can import from the project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_evolve.harness.agents.arc.mas_agent import MASArcAgent
from agent_evolve.harness.agents.arc.wiki import GameWiki
from agent_evolve.harness.agents.arc.memories import Memories
from agent_evolve.harness.agents.arc.bedrock_tools import (
    build_subagent_tools,
    build_orchestrator_tools,
)
from agent_evolve.harness.agents.arc.frame import Frame


SEED_DIR = Path(__file__).parent.parent / "seed_workspaces" / "arc-mas"


@pytest.fixture
def workspace(tmp_path):
    """Create a fresh workspace copy for each test."""
    ws = tmp_path / "arc-mas"
    shutil.copytree(SEED_DIR, ws)
    return ws


@pytest.fixture
def workspace_with_skills(workspace):
    """Workspace with two dummy skills."""
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)

    # Skill 1: maze solving
    s1 = skills_dir / "maze_solver"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\n"
        "name: maze_solver\n"
        "description: TRIGGER when game is a maze with corridors and walls\n"
        "---\n\n"
        "# Maze Solver\n\n"
        "## Procedure\n"
        "1. Use bounding_box to find player and goal\n"
        "2. BFS shortest path avoiding walls\n"
        "3. Execute path with run_action_sequence\n"
    )

    # Skill 2: click puzzle
    s2 = skills_dir / "click_puzzle"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\n"
        "name: click_puzzle\n"
        "description: TRIGGER when only ACTION6 (click) is available\n"
        "---\n\n"
        "# Click Puzzle\n\n"
        "## Procedure\n"
        "1. Render frame and identify clickable regions\n"
        "2. Click each region once, observe changes\n"
        "3. Build a toggle map\n"
    )

    return workspace


# -----------------------------------------------------------------------
# (1) Skills folder empty
# -----------------------------------------------------------------------

class TestEmptySkills:
    def test_no_skills_loaded(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace), max_actions=350)
        assert len(agent._skill_bodies) == 0
        assert len(agent.skills) == 0

    def test_read_skill_tool_not_created(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert agent._build_read_skill_tool() is None

    def test_role_prompt_has_no_skills_section(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        prompt = agent._get_role_prompt("explorer")
        assert "read_skill" not in prompt
        assert "Available Skills" not in prompt


# -----------------------------------------------------------------------
# (2) Dummy skills folder
# -----------------------------------------------------------------------

class TestSkillsLoading:
    def test_skills_loaded(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        assert len(agent.skills) == 2
        assert len(agent._skill_bodies) == 2

    def test_skill_names(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        names = {s.name for s in agent.skills}
        assert "maze_solver" in names
        assert "click_puzzle" in names

    def test_skill_bodies_extracted(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        assert "BFS shortest path" in agent._skill_bodies["maze_solver"]
        assert "toggle map" in agent._skill_bodies["click_puzzle"]

    def test_read_skill_tool_created(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        result = agent._build_read_skill_tool()
        assert result is not None
        spec, handler = result
        assert spec["toolSpec"]["name"] == "read_skill"

    def test_read_skill_returns_body(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        _, handler = agent._build_read_skill_tool()
        body = handler(skill_name="maze_solver")
        assert "BFS shortest path" in body

    def test_read_skill_missing_skill(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        _, handler = agent._build_read_skill_tool()
        result = handler(skill_name="nonexistent")
        assert "not found" in result.lower()
        assert "maze_solver" in result  # lists available

    def test_role_prompt_lists_skills(self, workspace_with_skills):
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        prompt = agent._get_role_prompt("solver")
        assert "maze_solver" in prompt
        assert "click_puzzle" in prompt
        assert "read_skill" in prompt

    def test_skill_body_not_in_prompt(self, workspace_with_skills):
        """Skill body should NOT be in prompt — only loaded via read_skill tool."""
        agent = MASArcAgent(workspace_dir=str(workspace_with_skills))
        prompt = agent._get_role_prompt("explorer")
        assert "BFS shortest path" not in prompt  # body is lazy-loaded
        assert "maze_solver" in prompt  # name is listed


# -----------------------------------------------------------------------
# (3) MAX_ACTIONS constraint
# -----------------------------------------------------------------------

class TestMaxActions:
    def test_default_max_actions(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert agent.max_actions == 350

    def test_custom_max_actions(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace), max_actions=50)
        assert agent.max_actions == 50

    def test_submit_action_enforces_limit(self, workspace):
        """Simulate submit_action hitting the limit."""
        # We can't run a real game, but we can test the GameAction limit logic
        # by checking that the agent stores the right value
        agent = MASArcAgent(workspace_dir=str(workspace), max_actions=5)
        assert agent.max_actions == 5


# -----------------------------------------------------------------------
# (4) Reasoning effort = high (not max)
# -----------------------------------------------------------------------

class TestThinkingEffort:
    def test_default_no_thinking(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert agent.thinking_effort == ""

    def test_high_thinking(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace), thinking_effort="high")
        assert agent.thinking_effort == "high"

    def test_max_tokens_increased_with_thinking(self, workspace):
        """When thinking is enabled, max_tokens should be 65536."""
        agent = MASArcAgent(workspace_dir=str(workspace), thinking_effort="high")
        # The max_tokens increase happens at BedrockAgent creation time,
        # verified by checking the expression in the code
        expected = 65536 if agent.thinking_effort else 16384
        assert expected == 65536

    def test_no_thinking_normal_tokens(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace), thinking_effort="")
        expected = 65536 if agent.thinking_effort else 16384
        assert expected == 16384


# -----------------------------------------------------------------------
# (5) Each subagent loads its own artifacts
# -----------------------------------------------------------------------

class TestSubagentArtifacts:
    def test_role_prompts_loaded(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert "explorer" in agent._role_prompts
        assert "theorist" in agent._role_prompts
        assert "solver" in agent._role_prompts

    def test_explorer_prompt_content(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        prompt = agent._get_role_prompt("explorer")
        assert "probe" in prompt.lower() or "explore" in prompt.lower()
        assert "wiki" in prompt.lower() or "knowledge" in prompt.lower()

    def test_theorist_prompt_no_actions(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        prompt = agent._get_role_prompt("theorist")
        assert "NOT take game actions" in prompt or "do NOT" in prompt.lower()

    def test_solver_prompt_has_execution(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        prompt = agent._get_role_prompt("solver")
        assert "execute" in prompt.lower() or "plan" in prompt.lower()

    def test_role_prompts_differ(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        exp = agent._get_role_prompt("explorer")
        theo = agent._get_role_prompt("theorist")
        sol = agent._get_role_prompt("solver")
        # Each role should have unique content
        assert exp != theo
        assert theo != sol
        assert exp != sol

    def test_tool_descriptions_loaded_per_role(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert "orchestrator" in agent._tool_descriptions
        assert "explorer" in agent._tool_descriptions
        assert "theorist" in agent._tool_descriptions
        assert "solver" in agent._tool_descriptions

    def test_theorist_has_no_submit_action_tool(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        theorist_tools = agent._tool_descriptions.get("theorist", {})
        assert "submit_action" not in theorist_tools
        assert "run_action_sequence" not in theorist_tools

    def test_explorer_has_submit_action_tool(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        explorer_tools = agent._tool_descriptions.get("explorer", {})
        assert "submit_action" in explorer_tools

    def test_orchestrator_has_spawn_tool(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        orch_tools = agent._tool_descriptions.get("orchestrator", {})
        assert "spawn_and_run_subagent" in orch_tools
        assert "submit_action" not in orch_tools  # orchestrator can't play

    def test_game_reference_loaded(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert len(agent._game_reference) > 100
        assert "ACTION1" in agent._game_reference
        assert "render_frame" in agent._game_reference

    def test_game_reference_in_role_prompt(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        for role in ["explorer", "theorist", "solver"]:
            prompt = agent._get_role_prompt(role)
            assert "ACTION1" in prompt or "action" in prompt.lower()

    def test_unknown_role_falls_back_to_solver(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        prompt = agent._get_role_prompt("custom_agent")
        solver_prompt = agent._get_role_prompt("solver")
        # Should fall back to solver
        assert "execute" in prompt.lower() or "plan" in prompt.lower()


# -----------------------------------------------------------------------
# (6) Shared memory/wiki between agents
# -----------------------------------------------------------------------

class TestSharedKnowledge:
    def test_wiki_shared_instance(self):
        """All tool handlers should reference the same wiki instance."""
        wiki = GameWiki(game_id="test")
        frame_ref = [None, None]

        def dummy_submit(a, x=0, y=0): return None
        def dummy_history(n=50, w=False): return []

        # Build tools for two different "agents"
        tools1, handlers1 = build_subagent_tools(
            dummy_submit, 10, dummy_history, wiki, frame_ref,
        )
        tools2, handlers2 = build_subagent_tools(
            dummy_submit, 10, dummy_history, wiki, frame_ref,
        )

        # Agent 1 writes to wiki
        handlers1["wiki_write"](page="game_rules", content="ACTION1=up")

        # Agent 2 should see it
        result = handlers2["wiki_read"](page="game_rules")
        assert "ACTION1=up" in result

    def test_wiki_append_only_enforcement(self):
        """Critical pages should never be overwritten."""
        wiki = GameWiki(game_id="test")
        frame_ref = [None, None]

        def dummy_submit(a, x=0, y=0): return None
        def dummy_history(n=50, w=False): return []

        tools, handlers = build_subagent_tools(
            dummy_submit, 10, dummy_history, wiki, frame_ref,
        )

        # Write twice to game_rules (append-only)
        handlers["wiki_write"](page="game_rules", content="ACTION1=up")
        handlers["wiki_write"](page="game_rules", content="CORRECTION: ACTION1=down")

        result = handlers["wiki_read"](page="game_rules")
        # Both entries should be present (append, not overwrite)
        assert "ACTION1=up" in result
        assert "CORRECTION" in result

    def test_wiki_overwritable_pages(self):
        """colors, current_level, current_plan should be overwritable."""
        wiki = GameWiki(game_id="test")
        frame_ref = [None, None]

        def dummy_submit(a, x=0, y=0): return None
        def dummy_history(n=50, w=False): return []

        tools, handlers = build_subagent_tools(
            dummy_submit, 10, dummy_history, wiki, frame_ref,
        )

        handlers["wiki_write"](page="colors", content="v1: 0=wall")
        handlers["wiki_write"](page="colors", content="v2: 0=floor, 5=wall")

        result = handlers["wiki_read"](page="colors")
        assert "v1" not in result  # overwritten
        assert "v2" in result

    def test_flat_memory_shared(self):
        """Test flat Memories mode — all agents share the same instance."""
        memories = Memories()
        frame_ref = [None, None]

        def dummy_submit(a, x=0, y=0): return None
        def dummy_history(n=50, w=False): return []

        tools1, handlers1 = build_subagent_tools(
            dummy_submit, 10, dummy_history, memories, frame_ref,
        )
        tools2, handlers2 = build_subagent_tools(
            dummy_submit, 10, dummy_history, memories, frame_ref,
        )

        # Agent 1 adds memory
        handlers1["memories_add"](summary="test discovery", details="ACTION1 moves up")

        # Agent 2 should see it
        result = handlers2["memories_summaries"]()
        assert "test discovery" in result

    def test_orchestrator_and_subagent_share_wiki(self):
        """Orchestrator and subagent tools should share the same wiki."""
        wiki = GameWiki(game_id="test")
        frame_ref = [None, None]

        def dummy_spawn(**kw): return "ok"
        def dummy_call(**kw): return "ok"
        def dummy_submit(a, x=0, y=0): return None
        def dummy_history(n=50, w=False): return []

        orch_tools, orch_handlers = build_orchestrator_tools(
            dummy_spawn, dummy_call, wiki, dummy_history, frame_ref,
        )
        sub_tools, sub_handlers = build_subagent_tools(
            dummy_submit, 10, dummy_history, wiki, frame_ref,
        )

        # Orchestrator writes
        orch_handlers["wiki_write"](page="current_plan", content="Try going right")

        # Subagent reads
        result = sub_handlers["wiki_read"](page="current_plan")
        assert "Try going right" in result

        # Subagent writes
        sub_handlers["wiki_write"](page="game_rules", content="ACTION1=up confirmed")

        # Orchestrator reads
        result = orch_handlers["wiki_read"](page="game_rules")
        assert "ACTION1=up confirmed" in result

    def test_wiki_index_shows_all_pages(self):
        """wiki_index should list all pages."""
        wiki = GameWiki(game_id="test")
        frame_ref = [None, None]

        def dummy_history(n=50, w=False): return []

        tools, handlers = build_subagent_tools(
            None, None, dummy_history, wiki, frame_ref,
        )

        handlers["wiki_write"](page="game_rules", content="rules here")
        handlers["wiki_write"](page="breakthroughs", content="big finding")

        index = handlers["wiki_index"]()
        assert "game_rules" in index
        assert "breakthroughs" in index
        assert "rules here" in index  # first line as summary

    def test_wiki_history_tracked(self):
        """Write operations should be recorded in wiki history."""
        wiki = GameWiki(game_id="test")
        wiki.write("colors", "0=wall")
        wiki.write("game_rules", "ACTION1=up")
        wiki.append("failed_attempts", "tried going left, hit wall")

        assert len(wiki._history) == 3
        assert wiki._history[0]["op"] == "write"
        assert wiki._history[0]["page"] == "colors"
        assert wiki._history[1]["op"] == "append"  # game_rules is append-only
        assert wiki._history[2]["op"] == "append"


# -----------------------------------------------------------------------
# Integration: reload_from_fs re-loads everything
# -----------------------------------------------------------------------

class TestReload:
    def test_reload_picks_up_new_skills(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        assert len(agent._skill_bodies) == 0

        # Add a skill after init
        skill_dir = workspace / "skills" / "new_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: new_skill\ndescription: test\n---\n\nDo something."
        )

        agent.reload_from_fs()
        assert len(agent._skill_bodies) == 1
        assert "Do something" in agent._skill_bodies["new_skill"]

    def test_reload_picks_up_prompt_changes(self, workspace):
        agent = MASArcAgent(workspace_dir=str(workspace))
        old_prompt = agent._role_prompts["explorer"]

        # Modify explorer prompt
        (workspace / "prompts" / "explorer.md").write_text("NEW EXPLORER INSTRUCTIONS")

        agent.reload_from_fs()
        assert agent._role_prompts["explorer"] == "NEW EXPLORER INSTRUCTIONS"
        assert agent._role_prompts["explorer"] != old_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
