#!/usr/bin/env python3
"""Play LS20 with VISION + batched code actions, then serve replay on port 7889.

Flow per LLM call:
1. SEE: LLM gets a PNG image of current grid + diff image from last batch
2. THINK: LLM reasons about what to try
3. ACT: LLM calls execute_actions(code) which runs multiple actions in one call
4. SEE: Gets new image showing result
"""

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent_evolve.harness.agents.arc.frame import Frame
from agent_evolve.harness.agents.arc.game_loop import convert_frame_data
from agent_evolve.harness.agents.arc.colors import COLOR_NAMES, COLOR_LEGEND, PALETTE_HEX
from agent_evolve.harness.agents.arc.grid_render import grid_to_image, image_to_base64, image_diff

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"ls20_{int(time.time())}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("play_ls20")

MAX_LLM_CALLS = 20
MAX_ACTIONS = 80


def img_block(grid):
    """Grid -> Bedrock image content block."""
    b64 = image_to_base64(grid_to_image(grid))
    return {"image": {"format": "png", "source": {"bytes": base64.b64decode(b64)}}}


def diff_block(grid_old, grid_new):
    """Diff image content block (red highlights)."""
    d = image_diff(grid_to_image(grid_old), grid_to_image(grid_new))
    if d is None:
        return None
    b64 = image_to_base64(d)
    return {"image": {"format": "png", "source": {"bytes": base64.b64decode(b64)}}}


def main():
    import arc_agi
    import boto3
    from arcengine import GameAction

    logger.info("=" * 60)
    logger.info("LS20 -- VISION + BATCHED ACTIONS -- max %d calls, %d actions", MAX_LLM_CALLS, MAX_ACTIONS)
    logger.info("=" * 60)

    arcade = arc_agi.Arcade()
    env = arcade.make("ls20", render_mode=None)
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    raw = env.reset()
    frame, meta = convert_frame_data(raw)

    frames: list[Frame] = [frame]
    captured: list[dict] = [{
        "step": 0, "action": "INIT",
        "grid": [list(r) for r in frame.grid],
        "levels_completed": meta.get("levels_completed", 0),
        "state": str(meta.get("state", "")),
        "level_changed": False,
    }]

    avail = meta.get("available_actions", [])
    action_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Build action executor that the LLM's code can call
    def step_action(action_name, x=-1, y=-1):
        """Execute one game action. Returns change summary string."""
        nonlocal action_count
        ga = GameAction.from_name(action_name)
        if ga.is_complex() and x >= 0 and y >= 0:
            ga.set_data({"x": min(x, 63), "y": min(y, 63)})

        prev_levels = meta.get("levels_completed", 0)
        prev_grid = frames[-1].grid

        r = env.step(ga)
        if isinstance(r, tuple):
            r = r[0]
        new_frame, new_meta = convert_frame_data(r)
        frames.append(new_frame)
        meta.update(new_meta)
        action_count += 1

        level_changed = meta.get("levels_completed", 0) > prev_levels
        captured.append({
            "step": len(captured), "action": action_name,
            "grid": [list(row) for row in new_frame.grid],
            "levels_completed": meta.get("levels_completed", 0),
            "state": str(meta.get("state", "")),
            "level_changed": level_changed,
        })

        diff = new_frame.change_summary(Frame(prev_grid))
        status = " LEVEL COMPLETE!" if level_changed else ""
        return f"#{action_count}: {action_name}{status} -> {diff}"

    # Shorthand helpers for the REPL
    def up(): return step_action("ACTION1")
    def down(): return step_action("ACTION2")
    def left(): return step_action("ACTION3")
    def right(): return step_action("ACTION4")
    def interact(): return step_action("ACTION5")
    def click(x, y): return step_action("ACTION6", x, y)
    def undo(): return step_action("ACTION7")

    system_prompt = f"""You are playing ARC-AGI-3 game LS20 (keyboard game, 7 levels).

Each turn you SEE the grid as a PNG image. You ACT by calling execute_actions() with Python code that takes multiple actions at once.

## Tools
- observe() -- Get a fresh image (FREE)
- execute_actions(code) -- Run Python code that calls action functions. Each action costs 1 from budget.
  Available functions in code:
    up()      -- ACTION1
    down()    -- ACTION2
    left()    -- ACTION3
    right()   -- ACTION4
    interact() -- ACTION5 (if available)
    click(x,y) -- ACTION6 (if available)
    undo()    -- ACTION7 (if available)
  Each returns a string describing what changed.
  Use print() to see results.

## Example
```
execute_actions(code=\"\"\"
# Try all 4 directions to learn mechanics
print(right())
print(down())
print(left())
print(up())
\"\"\")
```

## Strategy
1. First call: observe() to see the grid image
2. Study the image -- identify rooms, corridors, player, objects
3. Batch 3-8 actions per execute_actions() call based on your plan
4. Check the result image + diff image to see what happened
5. Adjust strategy and continue

Available actions: {', '.join(avail)}
Budget: {MAX_ACTIONS} actions
Colors: {COLOR_LEGEND}
"""

    messages: list[dict] = []
    llm_calls = 0

    # First message with initial image
    messages.append({
        "role": "user",
        "content": [
            img_block(frame.grid),
            {"text": (
                f"LS20 level 1. Available: {', '.join(avail)}. Budget: {MAX_ACTIONS} actions.\n"
                "Study this image, then start experimenting with execute_actions()."
            )},
        ],
    })

    tools = [
        {"toolSpec": {"name": "observe",
                       "description": "Get fresh image of current grid. FREE.",
                       "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
        {"toolSpec": {"name": "execute_actions",
                       "description": "Run Python code that calls action functions (up/down/left/right/interact/click/undo). Each action returns a change description. Use print() to see results. Multiple actions per call = efficient!",
                       "inputSchema": {"json": {"type": "object",
                                                 "properties": {"code": {"type": "string", "description": "Python code calling up(), down(), left(), right(), etc."}},
                                                 "required": ["code"]}}}},
        {"toolSpec": {"name": "reset_level",
                       "description": "Restart current level from scratch.",
                       "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}}}},
    ]

    while llm_calls < MAX_LLM_CALLS and action_count < MAX_ACTIONS:
        llm_calls += 1
        logger.info("--- LLM call #%d (actions: %d/%d) ---", llm_calls, action_count, MAX_ACTIONS)

        # Safe trim: always keep first message, trim pairs from middle
        if len(messages) > 20:
            # Keep first 2 messages (initial) and last 16
            messages = messages[:2] + messages[-16:]
            # Ensure alternating user/assistant
            while len(messages) > 2 and messages[2]["role"] != "user":
                messages.pop(2)

        try:
            resp = client.converse(
                modelId="us.anthropic.claude-opus-4-6-v1",
                system=[{"text": system_prompt}],
                messages=messages,
                toolConfig={"tools": tools},
                inferenceConfig={"maxTokens": 4000, "temperature": 0.3},
            )
        except Exception as e:
            logger.error("Bedrock error: %s", e)
            # Try to recover by trimming more aggressively
            messages = messages[:2] + messages[-4:]
            continue

        usage = resp.get("usage", {})
        total_input_tokens += usage.get("inputTokens", 0)
        total_output_tokens += usage.get("outputTokens", 0)

        content_blocks = resp.get("output", {}).get("message", {}).get("content", [])

        for b in content_blocks:
            if "text" in b:
                logger.info("LLM: %s", b["text"][:300])

        messages.append({"role": "assistant", "content": content_blocks})

        # Process tool calls
        has_tools = any("toolUse" in b for b in content_blocks)
        if not has_tools:
            messages.append({
                "role": "user",
                "content": [{"text": f"Call a tool! {MAX_ACTIONS - action_count} actions left. Use execute_actions() or observe()."}],
            })
            continue

        tool_results = []
        for block in content_blocks:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            name = tu["name"]
            inp = tu.get("input", {})
            tid = tu["toolUseId"]

            logger.info("Tool: %s", name)

            if name == "observe":
                tool_results.append({"toolResult": {"toolUseId": tid, "status": "success",
                    "content": [img_block(frames[-1].grid),
                                {"text": f"Level {meta.get('levels_completed',0)}/{meta.get('win_levels',0)} | Actions: {action_count}/{MAX_ACTIONS}"}]}})

            elif name == "reset_level":
                r = env.reset()
                if isinstance(r, tuple): r = r[0]
                f, m = convert_frame_data(r)
                frames.append(f); meta.update(m)
                captured.append({"step": len(captured), "action": "RESET",
                    "grid": [list(row) for row in f.grid],
                    "levels_completed": meta.get("levels_completed", 0),
                    "state": str(meta.get("state", "")), "level_changed": False})
                tool_results.append({"toolResult": {"toolUseId": tid, "status": "success",
                    "content": [img_block(f.grid), {"text": "Level reset."}]}})

            elif name == "execute_actions":
                code = inp.get("code", "")
                logger.info("Code:\n%s", code[:500])

                grid_before = frames[-1].grid
                actions_before = action_count

                # Execute the code in a namespace with action helpers
                import io, contextlib
                stdout_buf = io.StringIO()
                exec_ns = {
                    "up": up, "down": down, "left": left, "right": right,
                    "interact": interact, "click": click, "undo": undo,
                    "print": lambda *a, **k: print(*a, file=stdout_buf, **k),
                }
                try:
                    exec(code, exec_ns)
                except Exception as e:
                    stdout_buf.write(f"\nERROR: {e}")

                output = stdout_buf.getvalue()
                actions_this_call = action_count - actions_before
                logger.info("Code executed: %d actions, output:\n%s", actions_this_call, output[:500])

                # Build result: new image + diff image + text output
                result_content = [img_block(frames[-1].grid)]
                if actions_this_call > 0:
                    db = diff_block(grid_before, frames[-1].grid)
                    if db:
                        result_content.append(db)
                result_content.append({"text": (
                    f"Executed {actions_this_call} actions. Budget: {action_count}/{MAX_ACTIONS}\n"
                    f"Image 1: current state. Image 2: diff (red = all changes).\n"
                    f"Output:\n{output[:2000]}"
                )})
                tool_results.append({"toolResult": {"toolUseId": tid, "status": "success",
                    "content": result_content}})
            else:
                tool_results.append({"toolResult": {"toolUseId": tid, "status": "error",
                    "content": [{"text": f"Unknown tool: {name}"}]}})

        messages.append({"role": "user", "content": tool_results})

        if meta.get("levels_completed", 0) > 0:
            logger.info("*** LEVEL COMPLETED! ***")
            break

    # Summary
    logger.info("=" * 60)
    logger.info("DONE: %d LLM calls, %d actions, %d frames", llm_calls, action_count, len(captured))
    logger.info("Levels: %d/%d", meta.get("levels_completed", 0), meta.get("win_levels", 0))
    logger.info("Tokens: in=%d out=%d total=%d", total_input_tokens, total_output_tokens,
                total_input_tokens + total_output_tokens)
    logger.info("=" * 60)

    result = {
        "game_id": "ls20", "levels_completed": meta.get("levels_completed", 0),
        "total_levels": meta.get("win_levels", 0), "total_actions": action_count,
        "llm_calls": llm_calls, "game_completed": meta.get("levels_completed", 0) > 0,
        "usage": {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens,
                  "total_tokens": total_input_tokens + total_output_tokens},
    }
    with open(log_dir / "ls20_result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(log_dir / "ls20_frames.json", "w") as f:
        json.dump(captured, f)

    serve_replay(captured, result)


def serve_replay(frames_data, result):
    from flask import Flask, Response
    app = Flask(__name__)
    fj = json.dumps(frames_data); pj = json.dumps(PALETTE_HEX); nj = json.dumps(COLOR_NAMES)
    lv=result.get("levels_completed",0); tl=result.get("total_levels",0)
    ta=result.get("total_actions",0); lc=result.get("llm_calls",0)
    gc=result.get("game_completed",False)
    html=f"""<!DOCTYPE html><html><head><title>LS20 Replay</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Courier New',monospace;background:#1a1a2e;color:#eee;padding:20px}}h1{{color:#4FCC30;margin-bottom:6px}}.info{{color:#88D8F1;margin-bottom:12px;font-size:14px}}.container{{display:flex;gap:20px}}.grid-panel{{flex:0 0 auto}}.side-panel{{flex:1;min-width:350px;max-width:600px;display:flex;flex-direction:column}}canvas{{border:2px solid #333;display:block;cursor:crosshair}}.controls{{margin:12px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button{{background:#333;color:#eee;border:1px solid #555;padding:6px 14px;cursor:pointer;font-family:inherit;font-size:13px;border-radius:4px}}button:hover{{background:#444}}button.active{{background:#4FCC30;color:#000}}.slider-wrap{{flex:1;display:flex;align-items:center;gap:8px}}input[type=range]{{flex:1;accent-color:#4FCC30}}.step-label{{font-size:16px;font-weight:bold;min-width:80px}}.diff-box{{background:#0f0f23;padding:10px;border-radius:6px;font-size:12px;border-left:3px solid #F93C31;margin-bottom:10px;white-space:pre-wrap;max-height:120px;overflow-y:auto}}.coords{{background:#16213e;padding:6px 10px;border-radius:4px;font-size:12px;margin-bottom:10px}}.action-list{{flex:1;overflow-y:auto;background:#0f0f23;border-radius:6px;padding:6px;font-size:12px}}.action-item{{padding:4px 8px;cursor:pointer;border-radius:3px;margin-bottom:1px;display:flex;gap:6px;align-items:center}}.action-item:hover{{background:#1a1a3e}}.action-item.current{{background:#16213e;border-left:3px solid #4FCC30}}.action-item .sn{{color:#555;min-width:28px;text-align:right}}.action-item .ab{{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:bold;min-width:65px;text-align:center}}.action-item .lv{{color:#4FCC30;font-size:10px}}.legend{{display:flex;flex-wrap:wrap;gap:3px;margin-top:8px}}.legend-item{{display:flex;align-items:center;gap:3px;font-size:10px}}.legend-swatch{{width:12px;height:12px;border:1px solid #444}}.timeline{{display:flex;flex-wrap:wrap;gap:1px;margin-top:6px}}.tl-dot{{width:6px;height:6px;border-radius:50%;cursor:pointer}}.tl-dot.cur{{outline:2px solid #fff}}.tl-dot.lc{{outline:2px solid #4FCC30}}</style></head><body>
<h1>LS20 Replay (Vision + Batched Actions)</h1>
<div class="info">Levels: {lv}/{tl} | Actions: {ta} | LLM calls: {lc} | {"COMPLETE" if gc else "INCOMPLETE"}</div>
<div class="container"><div class="grid-panel"><canvas id="grid" width="512" height="512"></canvas>
<div class="controls"><button onclick="step(-1)">Prev</button><button id="playBtn" onclick="togglePlay()">Play</button><button onclick="step(1)">Next</button><div class="slider-wrap"><input type="range" id="slider" min="0" value="0" oninput="goTo(this.value)"><span class="step-label" id="stepLabel">0/0</span></div></div>
<div class="coords" id="coords">Hover over grid</div><div class="timeline" id="timeline"></div><div class="legend" id="legend"></div></div>
<div class="side-panel"><div id="actionBadge" class="action-badge" style="background:#888">INIT</div><div class="diff-box" id="diffBox">Initial state</div><h3 style="color:#88D8F1;font-size:13px;margin:8px 0 4px">All Actions ({ta})</h3><div class="action-list" id="actionList"></div></div></div>
<script>const frames={fj};const palette={pj};const colorNames={nj};const canvas=document.getElementById('grid'),ctx=canvas.getContext('2d');const slider=document.getElementById('slider');slider.max=frames.length-1;let cur=0,playing=false,playIv=null;const aCols={{'INIT':'#888','RESET':'#F93C31','ACTION1':'#1E93FF','ACTION2':'#4FCC30','ACTION3':'#FFDC00','ACTION4':'#FF851B','ACTION5':'#A356D6','ACTION6':'#E53AA3','ACTION7':'#88D8F1'}};const al=document.getElementById('actionList');frames.forEach((f,i)=>{{const d=document.createElement('div');d.className='action-item';d.id='ai-'+i;d.innerHTML='<span class="sn">#'+f.step+'</span><span class="ab" style="background:'+(aCols[f.action]||'#555')+';color:#fff">'+f.action+'</span>'+(f.level_changed?'<span class="lv">LEVEL UP</span>':'');d.onclick=()=>goTo(i);al.appendChild(d)}});const tl=document.getElementById('timeline');frames.forEach((f,i)=>{{const d=document.createElement('div');d.className='tl-dot'+(f.level_changed?' lc':'');d.style.background=aCols[f.action]||'#555';d.onclick=()=>goTo(i);tl.appendChild(d)}});const lg=document.getElementById('legend');palette.forEach((hex,i)=>{{const d=document.createElement('div');d.className='legend-item';d.innerHTML='<div class="legend-swatch" style="background:'+hex+'"></div>'+i+':'+colorNames[i];lg.appendChild(d)}});canvas.addEventListener('mousemove',e=>{{const r=canvas.getBoundingClientRect();const g=frames[cur].grid;const cw=canvas.width/g[0].length,ch=canvas.height/g.length;const x=Math.floor((e.clientX-r.left)/cw),y=Math.floor((e.clientY-r.top)/ch);if(x>=0&&x<g[0].length&&y>=0&&y<g.length)document.getElementById('coords').textContent='x='+x+' y='+y+' val='+g[y][x]+' ('+colorNames[g[y][x]]+')'}});function drawGrid(g){{const cw=canvas.width/g[0].length,ch=canvas.height/g.length;for(let y=0;y<g.length;y++)for(let x=0;x<g[y].length;x++){{ctx.fillStyle=palette[g[y][x]];ctx.fillRect(x*cw,y*ch,cw,ch)}}}}function diff(a,b){{if(!a||!b)return'';let ch=0,det=[];for(let y=0;y<a.length;y++)for(let x=0;x<a[y].length;x++)if(a[y][x]!==b[y][x]){{ch++;if(det.length<12)det.push('('+x+','+y+'): '+colorNames[a[y][x]]+'->'+colorNames[b[y][x]])}};if(!ch)return'No changes';return ch+' changed:\\n'+det.join('\\n')+(ch>12?'\\n...+'+(ch-12):'')}}function render(i){{drawGrid(frames[i].grid);slider.value=i;document.getElementById('stepLabel').textContent=i+'/'+(frames.length-1);document.getElementById('actionBadge').textContent=frames[i].action+(frames[i].level_changed?' LEVEL UP!':'');document.getElementById('actionBadge').style.background=aCols[frames[i].action]||'#555';document.getElementById('diffBox').textContent=i>0?diff(frames[i-1].grid,frames[i].grid):'Initial';al.querySelectorAll('.action-item').forEach((d,j)=>d.classList.toggle('current',j===i));const ai=document.getElementById('ai-'+i);if(ai)ai.scrollIntoView({{block:'nearest'}});tl.querySelectorAll('.tl-dot').forEach((d,j)=>d.classList.toggle('cur',j===i))}}function goTo(i){{cur=parseInt(i);render(cur)}}function step(d){{cur=Math.max(0,Math.min(frames.length-1,cur+d));render(cur)}}function togglePlay(){{playing=!playing;document.getElementById('playBtn').textContent=playing?'Pause':'Play';document.getElementById('playBtn').classList.toggle('active',playing);if(playing)playIv=setInterval(()=>{{if(cur>=frames.length-1){{togglePlay();return}}step(1)}},300);else clearInterval(playIv)}}document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(e.key===' '){{e.preventDefault();togglePlay()}}}});render(0);</script></body></html>"""
    @app.route("/")
    def index():
        return Response(html, content_type="text/html")
    logger.info("REPLAY: http://0.0.0.0:7889")
    app.run(host="0.0.0.0", port=7889, debug=False)


if __name__ == "__main__":
    main()
