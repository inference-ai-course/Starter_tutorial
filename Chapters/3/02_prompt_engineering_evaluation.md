# Part 2: Prompt Engineering and Evaluation

## Learning Objectives

- Craft effective prompts: system roles, clear constraints, structured outputs, few-shot examples, and reasoning variants
- Reduce hallucinations with guardrails and evaluate prompts for quality, stability, and parameter sensitivity (temperature, top_p) across backends
- Implement evaluation frameworks using both automated and human assessment methods
- Build production-ready prompt templates with proper validation and monitoring

## 2.1 Core Principles of Prompt Engineering

### System Prompts and Role Design

System prompts define the model's behavior, tone, and scope. They act as the foundation for consistent, reliable outputs.

#### Effective System Prompt Structure

```python
system_prompt = """
You are a [ROLE] with [EXPERTISE] specializing in [DOMAIN].

Your characteristics:
- [BEHAVIOR 1]
- [BEHAVIOR 2]
- [BEHAVIOR 3]

Your constraints:
- [CONSTRAINT 1]
- [CONSTRAINT 2]

Your output format:
- [FORMAT REQUIREMENT]
"""
```

#### Example: Technical Documentation Assistant

```python
system_prompt = """
You are a senior technical writer with 10 years of experience in software documentation.
You specialize in creating clear, concise, and accurate technical documentation for developers.

Your characteristics:
- Write in a professional but approachable tone
- Use active voice and present tense
- Include practical examples and code snippets
- Structure information logically with clear headings

Your constraints:
- Avoid jargon unless necessary (define when used)
- Keep sentences under 25 words when possible
- Provide step-by-step instructions for complex tasks
- Include troubleshooting sections for common issues

Your output format:
- Use Markdown formatting
- Include a brief overview section
- Structure content with hierarchical headings
- End with a summary or next steps section
"""
```

### Clear Instructions and Constraints

#### The CLEAR Framework

**C** - Context: Provide relevant background information
**L** - Length: Specify output length requirements
**E** - Examples: Show desired format and style
**A** - Audience: Define who will use the output
**R** - Requirements: List specific constraints and rules

#### Implementation Example

```python
def create_clear_prompt(task, context, length, examples, audience, requirements):
    return f"""
Context: {context}

Task: {task}

Requirements:
- Length: {length}
- Audience: {audience}
- {requirements}

Examples:
{examples}

Remember to follow all requirements and match the example format.
"""
```

### Few-Shot Learning and Reasoning

#### Few-Shot Pattern Design

```python
few_shot_prompt = """
Task: Generate product descriptions for e-commerce items.

Examples:

Input: Wireless Bluetooth Headphones
Output: Experience premium sound quality with our wireless Bluetooth headphones. 
Featuring active noise cancellation, 30-hour battery life, and comfortable 
over-ear design. Perfect for music lovers and professionals who demand 
exceptional audio performance.

Input: Stainless Steel Water Bottle
Output: Stay hydrated with our durable stainless steel water bottle. 
Double-wall vacuum insulation keeps drinks cold for 24 hours or hot for 12 hours. 
Leak-proof lid and sweat-free design make it perfect for gym, office, or outdoor adventures.

Input: Organic Cotton T-Shirt
Output:
"""
```

#### Reasoning Variants

**Chain-of-Thought**: Step-by-step reasoning
```python
cot_prompt = """
Solve this problem step by step, showing your reasoning process.

Problem: A train travels 120 miles in 2 hours. What is its average speed?

Solution:
Step 1: Identify given information
- Distance = 120 miles
- Time = 2 hours

Step 2: Recall the formula
Speed = Distance ÷ Time

Step 3: Calculate
Speed = 120 miles ÷ 2 hours = 60 miles per hour

Answer: The train's average speed is 60 miles per hour.
"""
```

**Preamble Pattern**: Concise reasoning for production
```python
preamble_prompt = """
Before providing your answer, give a one-line explanation of your approach starting with "Preamble:"

Preamble: I'll calculate speed using the formula distance ÷ time.

The train's average speed is 60 miles per hour.
"""
```

## 2.2 Guardrails Against Hallinations

### Provenance Validation

```python
class ProvenanceValidator:
    def __init__(self, source_documents):
        self.source_documents = source_documents
    
    def validate_claim(self, claim, source_text):
        """Check if a claim is supported by source text."""
        # Implement similarity checking
        # Use embeddings or keyword matching
        # Return confidence score and validation result
        
    def validate_response(self, response, sources):
        """Validate entire response against source documents."""
        claims = self.extract_claims(response)
        validation_results = []
        
        for claim in claims:
            supported = False
            for source in sources:
                if self.validate_claim(claim, source):
                    supported = True
                    break
            
            validation_results.append({
                'claim': claim,
                'supported': supported,
                'confidence': self.calculate_confidence(claim, sources)
            })
        
        return validation_results
```

### Trustworthiness Scoring

```python
class TrustworthinessScorer:
    def __init__(self, model_endpoint):
        self.model_endpoint = model_endpoint
    
    def score_response(self, response, context=None):
        """Calculate trustworthiness score for a response."""
        factors = {
            'consistency': self.check_consistency(response),
            'factual_accuracy': self.check_facts(response),
            'source_alignment': self.check_source_alignment(response, context),
            'logical_coherence': self.check_logic(response)
        }
        
        # Weighted scoring
        weights = {
            'consistency': 0.25,
            'factual_accuracy': 0.30,
            'source_alignment': 0.25,
            'logical_coherence': 0.20
        }
        
        score = sum(factors[key] * weights[key] for key in factors)\        \n        return {
            'score': score,
            'factors': factors,
            'recommendation': self.get_recommendation(score)
        }
    
    def get_recommendation(self, score):
        if score >= 0.8:
            return "High confidence - use as-is"
        elif score >= 0.6:
            return "Medium confidence - review recommended"
        else:
            return "Low confidence - verification required"
```

### Hallucination Detection Patterns

```python
def detect_potential_hallucinations(response):
    """Identify potential hallucination indicators."""
    indicators = {
        'excessive_specificity': len(re.findall(r'\d+', response)) > 10,
        'unverifiable_claims': any(phrase in response.lower() for phrase in [
            'studies show', 'research indicates', 'experts say'
        ]),
        'temporal_inconsistencies': check_temporal_consistency(response),
        'factual_contradictions': check_factual_consistency(response)
    }
    
    risk_score = sum(indicators.values()) / len(indicators)
    return {
        'risk_score': risk_score,
        'indicators': indicators,
        'needs_review': risk_score > 0.5
    }
```

## 2.3 Parameter Tuning for Stability and Style

### Temperature and Top_p Fundamentals

**Temperature (0.0 - 2.0)**:
- Controls randomness in token selection
- Lower values = more deterministic
- Higher values = more creative/varied

**Top_p (0.0 - 1.0)**:
- Controls cumulative probability threshold
- Lower values = more focused
- Higher values = more diverse

### Practical Parameter Ranges

```python
PARAMETER_PROFILES = {
    'accuracy_focused': {
        'temperature': 0.1,
        'top_p': 0.3,
        'description': 'High accuracy, low creativity'
    },
    'balanced': {
        'temperature': 0.5,
        'top_p': 0.7,
        'description': 'Balanced creativity and accuracy'
    },
    'creative': {
        'temperature': 0.8,
        'top_p': 0.9,
        'description': 'High creativity, varied outputs'
    },
    'deterministic': {
        'temperature': 0.0,
        'top_p': 0.1,
        'description': 'Maximum consistency'
    }
}
```

### Parameter Sensitivity Testing

```python
class ParameterSensitivityTester:
    def __init__(self, client, test_prompts):
n        self.client = client
        self.test_prompts = test_prompts
    
    def test_parameter_combinations(self, param_grid):
        """Test multiple parameter combinations."""
        results = []
        
        for temp, top_p in param_grid:
            combination_results = {
                'temperature': temp,
                'top_p': top_p,
                'outputs': [],
                'metrics': {}
            }
            
            for prompt in self.test_prompts:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    top_p=top_p
                )
                
                combination_results['outputs'].append({
                    'prompt': prompt,
                    'response': response.choices[0].message.content
                })
            
            # Calculate metrics for this combination
            combination_results['metrics'] = self.calculate_metrics(
                combination_results['outputs']
            )
            
            results.append(combination_results)
        
        return results
    
    def calculate_metrics(self, outputs):
        """Calculate stability and quality metrics."""
        responses = [output['response'] for output in outputs]
        
        return {
            'avg_length': np.mean([len(r) for r in responses]),
            'length_variance': np.var([len(r) for r in responses]),
            'lexical_diversity': self.calculate_lexical_diversity(responses),
            'consistency_score': self.calculate_consistency(responses)
        }
```

## 2.4 Evaluation Frameworks

### LLM-as-a-Judge Implementation

```python
class LLMJudge:
    def __init__(self, judge_model_client):
        self.judge_model = judge_model_client
    
    def evaluate_response(self, response, criteria, reference=None):
        """Evaluate a response using LLM-as-judge."""
        evaluation_prompt = f"""
You are an expert evaluator. Assess the following response based on these criteria:

Criteria: {criteria}

{f"Reference answer: {reference}" if reference else ""}

Response to evaluate: {response}

Provide scores (1-10) for each criterion and brief justifications:
- Relevance: [score] - [justification]
- Accuracy: [score] - [justification] 
- Clarity: [score] - [justification]
- Completeness: [score] - [justification]
- Style: [score] - [justification]

Overall score: [average]/10
"""
        
        evaluation = self.judge_model.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        
        return self.parse_evaluation(evaluation.choices[0].message.content)
    
    def batch_evaluate(self, responses, criteria):
        """Evaluate multiple responses efficiently."""
        evaluations = []
        
        for response in responses:
            evaluation = self.evaluate_response(response, criteria)
            evaluations.append(evaluation)
        
        return self.aggregate_evaluations(evaluations)
```

### Human-in-the-Loop Evaluation

```python
class HumanEvaluationFramework:
    def __init__(self, evaluation_interface):
        self.interface = evaluation_interface
    
    def create_evaluation_task(self, responses, criteria):
        """Create structured evaluation tasks for human reviewers."""
        tasks = []
        
        for i, response in enumerate(responses):
            task = {
                'task_id': f'eval_{i}',
                'response': response,
                'criteria': criteria,
                'questions': self.generate_evaluation_questions(criteria)
            }
            tasks.append(task)
        
        return tasks
    
    def generate_evaluation_questions(self, criteria):
        """Generate specific questions for each criterion."""
        questions = {}
        
        for criterion in criteria:
            if criterion == 'relevance':
                questions[criterion] = "How relevant is this response to the prompt? (1-5)"
            elif criterion == 'accuracy':
                questions[criterion] = "How accurate is the information provided? (1-5)"
            elif criterion == 'clarity':
                questions[criterion] = "How clear and understandable is the response? (1-5)"
        
        return questions
    
    def collect_feedback(self, task_results):
        """Collect and analyze human feedback."""
        aggregated_results = {
            'average_scores': {},
            'agreement_scores': {},
            'qualitative_feedback': []
        }
        
        for criterion in task_results[0]['scores']:
            scores = [result['scores'][criterion] for result in task_results]
            aggregated_results['average_scores'][criterion] = np.mean(scores)
            aggregated_results['agreement_scores'][criterion] = self.calculate_agreement(scores)
        
        return aggregated_results
```

## 2.5 Production-Ready Prompt Templates

### Template Management System

```python
class PromptTemplateManager:
    def __init__(self, template_dir):
        self.template_dir = template_dir
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Load templates from directory."""
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.template_dir, filename)) as f:
                    template = json.load(f)
                    self.templates[template['name']] = template
    
    def get_template(self, name, **kwargs):
        """Get template with variable substitution."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template = self.templates[name]
        prompt = template['template']
        
        # Substitute variables
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return prompt
    
    def validate_template(self, template):
        """Validate template structure and content."""
        required_fields = ['name', 'version', 'template', 'parameters']
        
        for field in required_fields:
            if field not in template:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate parameters
        for param in template['parameters']:
            if 'name' not in param or 'type' not in param:
                raise ValueError("Parameter must have 'name' and 'type'")
        
        return True
```

### Version Control and A/B Testing

```python
class PromptVersionControl:
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def create_version(self, template_name, template_content, metadata=None):
        """Create new version of a prompt template."""
        version_id = self.generate_version_id()
        
        version_data = {
            'template_name': template_name,
            'version_id': version_id,
            'content': template_content,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}\n        }\n        \n        self.storage.save_version(version_data)
        return version_id\n    \n    def deploy_version(self, template_name, version_id, traffic_percentage=100):\n        """Deploy a specific version with traffic control."""
        deployment = {\n            'template_name': template_name,\n            'version_id': version_id,\n            'traffic_percentage': traffic_percentage,\n            'deployed_at': datetime.now().isoformat()\n        }\n        \n        self.storage.save_deployment(deployment)\n        return deployment\n    \n    def run_ab_test(self, template_name, version_a, version_b, traffic_split=50):\n        """Set up A/B test between two versions."""
        test_config = {\n            'template_name': template_name,\n            'version_a': version_a,\n            'version_b': version_b,\n            'traffic_split': traffic_split,\n            'start_time': datetime.now().isoformat(),\n            'metrics': ['conversion_rate', 'user_satisfaction', 'response_quality']\n        }\n        \n        return test_config
```

## 2.6 Implementation Examples

### Example 1: Customer Support Bot

```python
class CustomerSupportBot:
    def __init__(self, model_client):
        self.model = model_client
        self.setup_prompts()
    
    def setup_prompts(self):
        self.system_prompt = """
You are a helpful customer support assistant for TechCorp.
You have access to order information, product details, and troubleshooting guides.

Guidelines:
- Be polite and professional
- Ask clarifying questions when needed
- Provide step-by-step solutions
- Escalate to human agent for complex issues
- Never make promises about refunds or compensation
"""
        
        self.few_shot_examples = [
            {
                "user": "My order hasn't arrived yet.",
                "assistant": "I'd be happy to help you track your order. Could you please provide your order number so I can check the status for you?"
            },
            {
                "user": "The product is broken.",
                "assistant": "I'm sorry to hear that. Can you describe what's wrong with the product? Also, when did you receive it? This will help me determine the best way to assist you."
            }
        ]
    
    def generate_response(self, user_message, order_context=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])
        
        # Add context if available
        if order_context:
            messages.append({
                "role": "system",
                "content": f"Context: {order_context}"
            })
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,  # Low temperature for consistency
            max_tokens=150
        )
        
        return response.choices[0].message.content
```

### Example 2: Content Quality Evaluator

```python
class ContentQualityEvaluator:
    def __init__(self, judge_model):
        self.judge = judge_model
        self.evaluation_criteria = {
            'clarity': 'Is the content clear and easy to understand?',
            'accuracy': 'Is the information factually correct?',
            'completeness': 'Does it cover all necessary aspects?',
            'engagement': 'Is the content engaging and interesting?',
            'seo_optimization': 'Is the content optimized for search engines?'
        }
    
    def evaluate_content(self, content, target_audience, purpose):
        evaluation_prompt = f"""
You are an expert content evaluator. Assess the following content based on these criteria:

Target Audience: {target_audience}
