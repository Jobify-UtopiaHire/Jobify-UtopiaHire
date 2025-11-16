from dotenv import load_dotenv
import asyncio
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import google, silero
from cv_parser import extract_text_from_cv
# Import the prompts and CV parser
from prompts import create_initial_prompts_en , create_initial_prompts_fr
from test_prompts import create_test_prompts, create_test_prompts_fr
import os
import requests
# Import the mailing module
from mailer import send_interview_report_email, extract_report_from_message
from livekit.agents import ConversationItemAddedEvent

# Load .env from parent directory (backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class Assistant(Agent):
    def __init__(self, agent_instruction) -> None:
        super().__init__(instructions=agent_instruction)
        self.interview_complete = False

# Global storage for session-specific data
# Maps room_name -> {cv_filename, job_description, candidate_email, candidate_name}
session_data = {}

BACKEND_SERVER = "http://localhost:8000"

async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the AI interview agent
    Gets CV filename, job description, and candidate info from backend server
    """
    
    try:
        room_name = ctx.room.name if ctx.room else None
        print(f"📋 Room name: {room_name}")
        
        cv_filename = None
        job_description = None
        candidate_email = None
        candidate_name = None
        job_title = "Position"  # Default
        
        # Try to fetch from backend server first
        if room_name:
            try:
                resp = requests.get(f"{BACKEND_SERVER}/api/session-data/{room_name}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    cv_filename = data.get('cv_filename')
                    job_description = data.get('job_description')
                    candidate_email = data.get('candidate_email')
                    candidate_name = data.get('candidate_name')
                    job_title = data.get('job_title', 'Position')
                    language = data.get('language')
                    print(f"✅ Got data from backend server")
            except Exception as e:
                print(f"⚠️ Could not fetch from server: {e}")
    
        if not cv_filename:
            print("❌ Error: No CV filename provided")
            return
        
        if not job_description:
            print("❌ Error: No job description provided")
            return
        
        print(f"✅ CV: {cv_filename}")
        print(f"✅ Job: {job_description[:100]}...")
        print(f"✅ Candidate: {candidate_name or 'Not provided'}")
        print(f"✅ Email: {candidate_email or 'Not provided'}")
        
        # Extract CV text
        print(f"📄 Reading CV: {cv_filename}")
        cv_file_path = os.path.join("..\\cv_uploads", cv_filename)
        
        if not os.path.exists(cv_file_path):
            print(f"❌ CV file not found: {cv_file_path}")
            return
        
        cv_text = extract_text_from_cv(cv_file_path)
        if not cv_text:
            cv_text = "No CV content extracted"

        # Generate personalized prompts
        print("🤖 Generating prompts...")
        # if language.lower() == "english":
        #     agent_instruction, session_instruction = create_initial_prompts_en(
        #         cv_text=cv_text,
        #         job_title=job_title,
        #         job_description_text=job_description,
        #     )
        # else:
        #     agent_instruction, session_instruction = create_initial_prompts_fr(
        #         cv_text=cv_text,
        #         job_title=job_title,
        #         job_description_text=job_description,
        #     )
        
        # ===== TEST MODE =====
        # Uncomment the lines below to enable fast test mode (2 questions only)
        # Comment out the production prompts above when testing
        # 
        if language.lower() == "english":
            agent_instruction, session_instruction = create_test_prompts(
                cv_text=cv_text,
                job_title=job_title,
                job_description_text=job_description,
            )
        else:
            agent_instruction, session_instruction = create_test_prompts_fr(
                cv_text=cv_text,
                job_title=job_title,
                job_description_text=job_description,
            )
        # =====================
        
        # Initialize session
        if language.lower() == "english":
            interview_language="en-US"
        else:
            interview_language="fr-FR"
        print("🎤 Initializing voice session...")
        session = AgentSession(
            vad=silero.VAD.load(min_silence_duration=2.5, min_speech_duration=0.5),
            llm=google.beta.realtime.RealtimeModel(voice="charon", language=interview_language)
        )
        assistant = Assistant(agent_instruction)
        
        # Start session
        await session.start(room=ctx.room, agent=assistant)

        print("✅ Starting interview...")
        await session.generate_reply(instructions=session_instruction)

        print("🎙️ Interview in progress...")
        @session.on("conversation_item_added")
        def on_conversation_item_added(event: ConversationItemAddedEvent):
            chat=event.item
            content=chat.content[0]
            print(content)
            if chat.role == "assistant":
                if "that concludes the interview" in content.lower():
                    print("🛑 Interview concluded by assistant.")
                    assistant.interview_complete = True
                    # Call async function in a synchronous callback
                    session.shutdown()
                    try:
                        asyncio.create_task(send_interview_report_email(
                            recipient_email=candidate_email,
                            candidate_name=candidate_name, 
                            job_title=job_title, 
                            report_text=content
                        ))
                    except Exception as e:
                        print(f"❌ Error sending email: {str(e)}")
                if "i'm terminating this interview immediately" in content.lower():
                    print("🛑 Interview terminated by assistant.")
                    assistant.interview_complete = True
                    session.shutdown()
                if "merci pour votre temps aujourd'hui" in content.lower():
                    print("🛑 Interview concluded by assistant.")
                    assistant.interview_complete = True
                    # Call async function in a synchronous callback
                    session.shutdown()
                    try:
                        asyncio.create_task(send_interview_report_email(
                            recipient_email=candidate_email,
                            candidate_name=candidate_name, 
                            job_title=job_title, 
                            report_text=content
                        ))
                    except Exception as e:
                        print(f"❌ Error sending email: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Make sure the upload folder exists (one level up from agent folder)
    cv_uploads_path = os.path.join("..", "cv_uploads")
    if not os.path.exists(cv_uploads_path):
        os.makedirs(cv_uploads_path)
        print("📁 Created 'cv_uploads' directory")

    print("=" * 50)
    print("🤖 AI Interview Coach Agent")
    print("=" * 50)
    print("Starting agent worker...")
    print("Waiting for connection from client...")
    print("=" * 50)
    
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))