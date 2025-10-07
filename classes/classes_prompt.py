import os
import uuid
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time
import ollama


# ===========================
# Question & Prompt Creation
# ===========================
class Question_Creation:
    def __init__(self, chat, prompt_template, return_motivation=False):
        self.chat = chat
        self.prompt_template = prompt_template
        self.return_motivation = return_motivation

    def extract_question(self, passage):
        """
        Sends the passage to the model and extracts a clean question,
        optionally also returning the motivation.
        """
        formatted_prompt = self.prompt_template.format(passage)
        response = self.chat.send_prompt(
            formatted_prompt,
            prompt_uuid=str(uuid.uuid4()),
            use_history=False,
            stream=False
        )

        # Split lines and strip whitespace
        lines = [line.strip() for line in response.raw_text.strip().split("\n") if line.strip()]
        
        if self.return_motivation:
            # Assume first line is question, second line is motivation
            question = lines[0]
            motivation = lines[1] if len(lines) > 1 else ""
            return question, motivation
        else:
            # Only return question
            question = lines[0]
            return question

class Prompt_Creation_Dual:
    def __init__(self, fewshot_examples):
        self.fewshot_examples = fewshot_examples

    def create_prompt_base(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, generate **ONE clear, concise question** that meaningfully connects both passages.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt

    def create_prompt_inpair(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below to see the difference between good and bad questions.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt


    def create_prompt_inpair_motivation(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below, which include bad questions and explanations describing why they are inadequate.\n"
            "**For new passages, only output the good question—do not write explanations.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n"
                f"Why the bad question is inadequate: {ex['motivation']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt



# ===========================
# Question & Prompt Creation
# ===========================
class Question_Creation:
    def __init__(self, chat, prompt_template, return_motivation=False):
        self.chat = chat
        self.prompt_template = prompt_template
        self.return_motivation = return_motivation

    def extract_question(self, **passages):
        """
        Sends the passage(s) to the model and extracts a clean question,
        optionally also returning the motivation.
        """
        # Safely format prompt with named placeholders
        formatted_prompt = self.prompt_template.format_map(passages)

        response = self.chat.send_prompt(
            formatted_prompt,
            prompt_uuid=str(uuid.uuid4()),
            use_history=False,
            stream=False
        )

        lines = [line.strip() for line in response.raw_text.strip().split("\n") if line.strip()]

        # Safe fallback if model returns nothing
        if not lines:
            if self.return_motivation:
                return "No question available", ""
            else:
                return "No question available"

        if self.return_motivation:
            question = lines[0]
            motivation = lines[1] if len(lines) > 1 else ""
            return question, motivation
        else:
            return lines[0]


class Prompt_Creation:
    def __init__(self, documents, fewshot_examples):
        if len(documents) != len(fewshot_examples):
            raise ValueError("Documents and few-shot examples must have the same length")
        self.documents = documents
        self.examples = fewshot_examples

    # ----------------------------
    # Strict Base Prompt
    # ----------------------------
    def create_prompt_base(self):
        prompt_lines = [
            "You are a strict question generator. Generate ONLY ONE Good Question from the given Document.",
            "Output only the question. No explanations, no extra text, no punctuation beyond the question itself.\n"
        ]
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")
        prompt_lines.append("Example to predict:")
        prompt_lines.append("Document: {passage}")  # named placeholder
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)

    # ----------------------------
    # Strict In-Pair Prompt
    # ----------------------------
    def create_prompt_inpair(self):
        prompt_lines = [
            "You are a strict question generator. Generate ONLY ONE Good Question from the given Document.",
            "Do NOT generate bad questions. Output ONLY the Good Question.",
            "Output only the question. No explanations, no extra text, no punctuation beyond the question itself.\n"
        ]
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            if "bad_question" in ex:
                prompt_lines.append(f"Bad Question: {ex['bad_question'].strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")
        prompt_lines.append("Example to predict:")
        prompt_lines.append("Document: {passage}")
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)

    # ----------------------------
    # Strict In-Pair with Motivation
    # ----------------------------
    def create_prompt_inpair_motivation(self):
        """
        Creates a robust prompt that generates:
        - ONLY ONE Good Question (answerable from the Document)
        - Optional Motivation: why a Bad Question is not ideal (only for few-shot examples)
        """
        prompt_lines = [
            "You are a precise and strict question-generation assistant.",
            "For the given document, generate exactly ONE Good Question.",
            "Output ONLY the question. Do NOT include explanations, extra text, or formatting.",
            "If the document is empty or no meaningful question can be generated, output 'No question available'.\n"
        ]

        # Few-shot examples
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            if "bad_question" in ex:
                prompt_lines.append(f"Bad Question: {ex['bad_question'].strip()}")
            if "motivation" in ex:
                prompt_lines.append(f"Motivation: {ex['motivation'].strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")

        # Prediction instruction
        prompt_lines.append("Now generate a question for the following document:")
        prompt_lines.append("Document: {passage}")
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)

class Prompt_Creation_Dual:
    def __init__(self, fewshot_examples):
        self.fewshot_examples = fewshot_examples

    def create_prompt_base(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, generate **ONE clear, concise question** that meaningfully connects both passages.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt

    def create_prompt_inpair(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below to see the difference between good and bad questions.\n"
            "**Only output the question—do not add explanations or extra text.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt


    def create_prompt_inpair_motivation(self):
        prompt = (
            "You are a precise question-generation assistant.\n"
            "Given **two passages**, write **ONE high-quality question** that connects both passages.\n"
            "Use the examples below, which include bad questions and explanations describing why they are inadequate.\n"
            "**For new passages, only output the good question—do not write explanations.**\n\n"
        )
        for ex in self.fewshot_examples:
            prompt += (
                f"Passage 1: {ex['passage1']}\n"
                f"Passage 2: {ex['passage2']}\n"
                f"Good question: {ex['query']}\n"
                f"Bad question: {ex['bad_query']}\n"
                f"Why the bad question is inadequate: {ex['motivation']}\n\n"
            )
        prompt += (
            "Now generate a question for the following passages:\n"
            "Passage 1: {passage1}\n"
            "Passage 2: {passage2}\n"
            "Question:"
        )
        return prompt


class Prompt_Creation_Correct:
    def __init__(self, documents, fewshot_examples):
        if len(documents) != len(fewshot_examples):
            raise ValueError("Documents and few-shot examples must have the same length")
        self.documents = documents
        self.examples = fewshot_examples

    # ----------------------------
    # Strict Base Prompt
    # ----------------------------
    def create_prompt_base(self):
        prompt_lines = [
            "You are a strict question generator. Generate ONLY ONE Good Question from the given Document changing the formulation of the concept.",
            "Output only the question. No explanations, no extra text, no punctuation beyond the question itself.\n"
        ]
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")
        prompt_lines.append("Example to predict:")
        prompt_lines.append("Document: {passage}")  # named placeholder
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)

    # ----------------------------
    # Strict In-Pair Prompt
    # ----------------------------
    def create_prompt_inpair(self):
        prompt_lines = [
            "You are a strict question generator. Generate ONLY ONE Good Question from the given Document changing the formulation of the concept.",
            "Do NOT generate bad questions. Output ONLY the Good Question.",
            "Output only the question. No explanations, no extra text, no punctuation beyond the question itself.\n"

        ]
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            if "bad_question" in ex:
                prompt_lines.append(f"Bad Question: {ex['bad_question'].strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")
        prompt_lines.append("Example to predict:")
        prompt_lines.append("Document: {passage}")
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)

    # ----------------------------
    # Strict In-Pair with Motivation
    # ----------------------------
    def create_prompt_inpair_motivation(self):
        """
        Creates a robust prompt that generates:
        - ONLY ONE Good Question (answerable from the Document)
        - Optional Motivation: why a Bad Question is not ideal (only for few-shot examples)
        """
        prompt_lines = [
            "You are a strict question generator. Generate ONLY ONE Good Question from the given Document changing the formulation of the concept.",
            "For the given document, generate exactly ONE Good Question.",
            "Output ONLY the question. Do NOT include explanations, extra text, or formatting.",
            "If the document is empty or no meaningful question can be generated, output 'No question available'.\n"

        ]

        # Few-shot examples
        for idx, (doc, ex) in enumerate(zip(self.documents, self.examples), start=1):
            prompt_lines.append(f"Example {idx}:")
            prompt_lines.append(f"Document: {doc.strip()}")
            if "bad_question" in ex:
                prompt_lines.append(f"Bad Question: {ex['bad_question'].strip()}")
            if "motivation" in ex:
                prompt_lines.append(f"Motivation: {ex['motivation'].strip()}")
            prompt_lines.append(f"Good Question: {ex['good_question'].strip()}")
            prompt_lines.append("")

        # Prediction instruction
        prompt_lines.append("Now generate a question for the following document:")
        prompt_lines.append("Document: {passage}")
        prompt_lines.append("Good Question:")
        return "\n".join(prompt_lines)


