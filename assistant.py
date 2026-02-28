
from openai import OpenAI
from pathlib import Path


class PetalAssitant:

    def __init__(self, api_key) -> None:
        self.api_key = api_key

        self.client = self.get_client()
        # self.assistant, self.thread = self.create_assistant()

    def create_openai_assistant_prompt(self, user_data):

        # Constructing a detailed prompt based on the comprehensive user data collected
        prompt = f"""
        User Profile:
        - Age: {user_data['age']}
        - Gender: {user_data['gender']}
        - General Health: {user_data['health_status']}
        - user Description on their emotion: {user_data.get('description', "Not provided")}
        
        User Needs:
        """

        prompt += f"\nTask:\n"
        prompt += "- Provide detailed advice on mental considering the user's health, current medication, past experiences, and priorities.\n"
        prompt += f"- Include personalized learning resources based on the user's preferred learning method: {user_data.get('learning_preferences', 'No preference')}.\n"

        return prompt


    def get_client(self):
        return OpenAI(api_key=self.api_key)
    
    def create_assistant(self):
        assistant = self.client.beta.assistants.create(
            name="Petal Health Agent",
            instructions="You are an expert Psychiatrists. Your are an assistant \
                that offers empathetic support and self-help resources, ensures \
                privacy and confidentiality, recognizes and manages crises by directing users \
                to emergency services, give medical advice, actively listens and responds \
                compassionately, regularly updates its knowledge and functionality based on \
                user feedback, and utilizes advanced NLP(Natural Language Processing: for understanding\
                and generating human like responses) and sentiment analysis for effective \
                interaction. Use this information to provide tailored advice on mental illnesses",
            tools=[{"type": "retrieval"}],
            model="gpt-4o-mini",
        )
        thread = self.client.beta.threads.create()
        return assistant, thread


    def generate_response(self, user_data, api_key):

        prompt = self.create_openai_assistant_prompt(user_data)

        response = self.client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "assistant", "content": prompt}])
        
        return response.choices[0].message.content

    
    def chat(self, messages):
        response = self.client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        return response.choices[0].message.content
    
    # def get_clinics(self, zip_code):
    #     prompt = f"Povide longitude and latitude of local clinics in the area of the zipcode of {zip_code} in python dictionary format only. Do not provide any other text. No intro text. No description. "
    #     prompt += '''Here is an example for the format:  
    #                 { 
    #                 "clinics": [
    #                     {
    #                         "name": "Health Care Clinic",
    #                         "lat": 37.7749,
    #                         "lon": -122.4194,
    #                         "services_offered": ["General Health", "Specialty Care"],
    #                     }
    #                 ]}
    #              '''
    #     response = self.client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "assistant", "content": prompt}])

    #     return response.choices[0].message.content

    def text_to_speech(self, text, voice):
        speech_file_path = Path("audio.mp3")
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        response.stream_to_file(speech_file_path)

    def transcribe(self, audio_path):
        audio_file= open(audio_path, "rb")

        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )

        return transcription.text


