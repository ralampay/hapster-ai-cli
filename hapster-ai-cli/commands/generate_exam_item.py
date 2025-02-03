#from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
import json

# Define a response schema
response_schema = [
    ResponseSchema(name="question", description="The item's question"),
    ResponseSchema(name="choices", description="Dictionary of choices with letters A, B, C, D"),
    ResponseSchema(name="correct_answer", description="The letter corresponding to the correct answer")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# Get the format instructions for the prompt
format_instructions = output_parser.get_format_instructions()

# Define the prompt template with explicit JSON format instructions
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="""
    Create a multiple-choice question about {topic}. The response must be in JSON format with the following structure:
    {{
        "question": "The question text",
        "choices": {{
            "A": "Option 1",
            "B": "Option 2",
            "C": "Option 3",
            "D": "Option 4"
        }},
        "correct_answer": "The letter of the correct option (A, B, C, or D)"
    }}

    Do not include any additional text or explanations. Only return the JSON object.

    Example:
    {{
        "question": "What is the capital of France?",
        "choices": {{
            "A": "Paris",
            "B": "London",
            "C": "Berlin",
            "D": "Madrid"
        }},
        "correct_answer": "A"
    }}

    {format_instructions}
    """
)

class GenerateExamItem:
    def __init__(self, topic="AWS Solutions Architect", model_id="gpt-3.5-turbo", openai_api_key=None):
        self.model_id = model_id

        self.llm = ChatOpenAI(
            model=self.model_id
        )

        self.topic = topic

        self.chain = prompt_template | self.llm

    def execute(self):
        response = self.chain.invoke({ "topic": self.topic, "format_instructions": format_instructions })

        parsed_response = output_parser.parse(response.content)

        print(json.dumps(parsed_response, indent=4))
