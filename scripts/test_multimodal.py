import asyncio
import logging
import os
import sys

# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.agents.multimodal import (
    VisionAgent,
    AudioAgent,
    SensorFusionAgent,
    ARInsightAgent,
    AgentOrchestrator,
)

async def main():
    logging.basicConfig(level=logging.INFO)
    agents = [VisionAgent(), AudioAgent(), SensorFusionAgent(), ARInsightAgent()]
    orchestrator = AgentOrchestrator()
    orchestrator.register_agents(agents)

    ok = await orchestrator.initialize_all(parallel=True)
    print("initialize_all:", ok)

    res1 = await orchestrator.execute_for_agent("VisionAgent", {"action":"analyze","params":{"image_path":"sample.jpg"}})
    print("VisionAgent analyze:", res1)

    res2 = await orchestrator.execute_for_agent("AudioAgent", {"action":"transcribe","params":{"audio_path":"sample.wav"}})
    print("AudioAgent transcribe:", res2)

    res3 = await orchestrator.execute_for_agent("SensorFusionAgent", {"action":"fuse","params":{"modalities":["vision","audio"]}})
    print("SensorFusionAgent fuse:", res3)

    res4 = await orchestrator.execute_for_agent("ARInsightAgent", {"action":"generate_overlay","params":{"scene_data":{}}})
    print("ARInsightAgent overlay:", res4)

    # shutdown
    for a in agents:
        await a.shutdown()
    print("shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
