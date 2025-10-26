import os
import httpx
from uagents import Agent, Context
from uagents.protocol import ProtocolManifest
from uagents.setup import fund_agent_if_low

from protocol import tutor_proto_v1
from models import (
    Slide,
    GetCourseRequest, GetCourseResponse,
    AnswerStepRequest, AnswerStepResponse,
    TutorInfoRequest, TutorInfoResponse,
    ErrorMessage,
)

# ====== Config via env ======
AGENT_SEED = os.getenv("AGENT_SEED", "studysnaps_render_seed_v1")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://<your-backend>.onrender.com")
PUBLIC_ENDPOINT = os.getenv("PUBLIC_ENDPOINT", "https://<this-agent-service>.onrender.com/submit")

# Render provides PORT automatically
PORT = int(os.getenv("PORT", "8000"))

# ====== Agent ======
agent = Agent(
    name="studysnaps_tutor_agent",
    seed=AGENT_SEED,
    port=PORT,                        # Bind to Render's PORT
    endpoint=[PUBLIC_ENDPOINT],       # uAgents inbox (must end with /submit)
)
agent.include(tutor_proto_v1, publish_manifest=False)

http_client = httpx.AsyncClient(timeout=30.0)

# ====== Handlers (proxy to your existing backend) ======
@tutor_proto_v1.on_message(model=TutorInfoRequest, replies={TutorInfoResponse, ErrorMessage})
async def info(ctx: Context, sender: str, msg: TutorInfoRequest):
    try:
        r = await http_client.get(f"{BACKEND_BASE_URL}/get-tutor-info")
        r.raise_for_status()
        await ctx.send(sender, TutorInfoResponse(tutor_id=r.json().get("tutor_id", "ERROR")))
    except Exception as e:
        await ctx.send(sender, ErrorMessage(error=f"/get-tutor-info failed: {e}"))

@tutor_proto_v1.on_message(model=GetCourseRequest, replies=GetCourseResponse)
async def course(ctx: Context, sender: str, msg: GetCourseRequest):
    try:
        r = await http_client.get(f"{BACKEND_BASE_URL}/get-course")
        r.raise_for_status()
        slides = [Slide(**s) for s in r.json()]
        await ctx.send(sender, GetCourseResponse(slides=slides))
    except Exception:
        await ctx.send(sender, GetCourseResponse(slides=[]))

@tutor_proto_v1.on_message(model=AnswerStepRequest, replies=AnswerStepResponse)
async def answer(ctx: Context, sender: str, msg: AnswerStepRequest):
    try:
        payload = {"image_b64": msg.image_b64, "filename": "from_agent.png"}
        headers = {"X-Session-Id": msg.session_id}
        r = await http_client.post(f"{BACKEND_BASE_URL}/answer_base64", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        await ctx.send(sender, AnswerStepResponse(
            answer_value=data.get("answer_value", "Error: No answer"),
            solving_completed=data.get("solving_completed", False),
        ))
    except Exception as e:
        await ctx.send(sender, AnswerStepResponse(
            answer_value=f"Proxy error: {e}",
            solving_completed=False,
        ))

# ====== Optional: register on Fetch Almanac (discovery) ======
@agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Agent address: {agent.address}")
    ctx.logger.info(f"Wallet: {agent.wallet.address()}")
    # If you want it discoverable on-chain:
    try:
        fund_agent_if_low(agent.wallet.address())           # free faucet test tokens
        manifest = ProtocolManifest.from_protocol(tutor_proto_v1)
        await agent.publish_manifest(manifest)
        ctx.logger.info("Manifest published to Almanac.")
    except Exception as e:
        ctx.logger.error(f"Publish failed: {e} (agent still works by address)")

@agent.on_event("shutdown")
async def shutdown(ctx: Context):
    await http_client.aclose()

if __name__ == "__main__":
    agent.run()
