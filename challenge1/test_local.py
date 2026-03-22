"""
Test the agent locally by simulating a /solve request.
Usage: python test_local.py "Create an employee named Ola Nordmann with email ola@test.no"
"""

import asyncio
import sys
from dotenv import load_dotenv
load_dotenv("../.env")

from agent import TripletexAgent

# Sandbox credentials — get from https://app.ainm.no/
SANDBOX_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3Njg2NDAzLCJ0b2tlbiI6ImVkMGIwOGZkLThjMzItNDdkNy1iY2Y4LWVkYzRkODMzMWZkYSJ9"


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_local.py '<prompt>'")
        print('Example: python test_local.py "Opprett en ansatt med navn Ola Nordmann og e-post ola@test.no"')
        sys.exit(1)

    prompt = sys.argv[1]
    token = SANDBOX_TOKEN or input("Enter sandbox session token: ").strip()

    if not token:
        print("Error: need a session token")
        sys.exit(1)

    print(f"Prompt: {prompt}")
    print(f"Base URL: {SANDBOX_BASE_URL}")
    print("---")

    agent = TripletexAgent(base_url=SANDBOX_BASE_URL, session_token=token)
    await agent.solve(prompt, files=[])

    print("---")
    print(f"Total API calls: {agent.client.call_count}")
    print(f"Total errors: {agent.client.error_count}")


if __name__ == "__main__":
    asyncio.run(main())
