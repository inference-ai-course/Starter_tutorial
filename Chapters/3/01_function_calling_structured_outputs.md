# Part 1: Function Calling and Structured Outputs

## Learning Objectives

- Understand function/tool calling: structured functions (JSON Schema) vs custom tools (free-form text) and when to use each
- Produce reliable, machine-consumable responses with JSON Schema, grammar/regex constraints
- Validate and evaluate output consistency across providers (OpenAI-compatible) and local endpoints (e.g., HuggingFace, Ollama)
- Implement cross-platform solutions using open-source tools instead of proprietary platforms

## 1.1 What is Function (Tool) Calling?

Function calling (also known as tool calling) enables AI models to interact with external systems by selecting and invoking predefined functions with structured arguments. This paradigm transforms AI models from passive text generators into active agents that can perform actions and retrieve information.

### The Function Calling Loop

```
User Prompt → Model Analysis → Function Selection → Argument Generation → Function Execution → Result Return → Final Answer
```

**Benefits:**
- **Structured arguments**: Predictable, validated inputs
- **Determinism**: Consistent behavior across calls
- **Safer integrations**: Type-safe parameter passing
- **Reproducibility**: Same inputs produce same outputs

### Two Types of Tools

#### 1. Function Tools (JSON Schema)
- **Predictable arguments** with strict typing
- **Validation** at parameter level
- **Precise parsing** with error handling
- **Best for**: APIs, databases, calculations

#### 2. Custom Tools (Free-form)
- **Raw text payloads** (SQL, scripts, configs)
- **Flexible format** when JSON is too rigid
- **Domain-specific languages** (DSLs)
- **Best for**: Complex queries, custom formats, legacy systems

## 1.2 Designing Tools with JSON Schema

JSON Schema provides a standardized way to define function parameters, ensuring consistency and enabling validation.

### Basic Schema Structure

```json
{
  "name": "function_name",
  "description": "Clear description of what the function does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "Description of parameter 1"
      },
      "param2": {
        "type": "integer",
        "minimum": 0,
        "maximum": 100
      }
    },
    "required": ["param1"],
    "additionalProperties": false
  }
}
```

### Schema Best Practices

1. **Use descriptive names**: `calculate_mortgage_payment` vs `calc`
2. **Provide clear descriptions**: Help the model understand when to use each function
3. **Define constraints**: Use `minimum`, `maximum`, `enum`, `pattern`
4. **Mark required fields**: Explicitly specify required parameters
5. **Disallow additional properties**: Prevent unexpected parameters

### Example: Weather API Tool

```python
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates (lat,lon)"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        }
    }
}
```

## 1.3 Structured Outputs Beyond Tools

### JSON Mode vs Schema-Constrained Outputs

**JSON Mode**: Model can generate any valid JSON
- **Pros**: Flexible, easy to implement
- **Cons**: Unpredictable structure, may be empty or malformed

**Schema-Constrained**: Enforce specific JSON structure
- **Pros**: Predictable output, validation-ready
- **Cons**: More complex setup, potential creativity reduction

### Advanced Constraints

#### Grammar Constraints (EBNF/CFG)
Constrain token-level generation to specific grammars:
- Date formats: `YYYY-MM-DD`
- List structures: Comma-separated values
- Domain-specific languages: SQL, regex patterns

#### Regex Constraints
Constrain primitive outputs:
- Phone numbers: `^\d{3}-\d{3}-\d{4}$`
- Email addresses: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- Product codes: `^[A-Z]{2}\d{4}$`

#### Choice Lists
Enumerate valid outputs:
- Sentiment: `["positive", "negative", "neutral"]`
- Categories: `["bug", "feature", "enhancement"]`
- Ratings: `["1", "2", "3", "4", "5"]`

## 1.4 Reliability Tactics

### Tool Allowlists
Restrict which tools can be used to improve determinism and safety:
```python
allowed_tools = ["get_weather", "get_news"]  # Explicit allowlist
```

### Preamble Reasoning
Ask the model to explain tool selection for observability:
```python
system_prompt = """
Before calling any tool, provide a one-line explanation starting with "Preamble:"
Example: "Preamble: I need to get weather data for the user's location"
"""
```

### Consistency Pitfalls and Solutions

**Problem**: Models vary in output format and structure
**Solutions**:
- Use schema-based prompting
- Implement post-parse validators
- Parse only JSON blocks from mixed responses
- Standardize field ordering and naming

### Error Handling Patterns

```python
def safe_json_parse(response_text):
    """Extract and parse JSON from potentially mixed responses."""
    try:
        # Try direct parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Extract JSON from code blocks
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try to find JSON-like content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError("No valid JSON found in response")
```

## 1.5 Cross-Provider and Local Endpoints

### OpenAI-Compatible APIs
Most providers support OpenAI's API format, enabling code reuse:

```python
# Works with OpenAI, HuggingFace, Ollama, vLLM
from openai import OpenAI

client = OpenAI(
    api_key="your_key",
    base_url="https://api.provider.com/v1"  # Change base_url for different providers
)
```

### HuggingFace Inference API
Open-source alternative to proprietary platforms:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="microsoft/DialoGPT-medium",
    token="your_hf_token"
)

response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

### Local Endpoint Options

#### Ollama
- **Pros**: Easy setup, good for development
- **Cons**: Limited model selection, moderate performance

#### vLLM
- **Pros**: High performance, production-ready
- **Cons**: Complex setup, resource-intensive

### Provider Comparison Framework

```python
def evaluate_provider(client, test_prompts, schema):
    """Evaluate provider performance across multiple metrics."""
    results = {
        'valid_json_rate': 0,
        'schema_compliance': 0,
        'field_completeness': 0,
        'avg_latency': 0,
        'token_usage': 0
    }
    
    for prompt in test_prompts:
        start_time = time.time()
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object", "schema": schema}
            )
            latency = time.time() - start_time
            
            # Validate response
            json_data = json.loads(response.choices[0].message.content)
            if validate_schema(json_data, schema):
                results['schema_compliance'] += 1
            results['valid_json_rate'] += 1
            
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
    
    # Calculate averages
    n = len(test_prompts)
    for key in results:
        if key not in ['avg_latency', 'token_usage']:
            results[key] = (results[key] / n) * 100
    
    return results
```

## 1.6 Implementation Examples

### Example 1: Coffee Recipe Generator

```python
import json
from typing import Dict, Any

def make_coffee(coffee_type: str) -> Dict[str, Any]:
    """Generate a coffee recipe based on type."""
    recipes = {
        "espresso": {
            "coffee_grams": 18,
            "water_ml": 36,
            "brew_time_seconds": 25,
            "temperature_celsius": 93
        },
        "cappuccino": {
            "coffee_grams": 18,
            "water_ml": 36,
            "milk_ml": 120,
            "milk_foam": "thick",
            "brew_time_seconds": 25
        }
    }
    return recipes.get(coffee_type, {"error": "Recipe not found"})

def random_coffee_fact() -> Dict[str, Any]:
    """Return a random coffee fact."""
    facts = [
        {"fact": "Coffee was first discovered in Ethiopia around 850 AD"},
        {"fact": "Espresso means 'pressed out' in Italian"},
        {"fact": "Coffee is the world's second-most traded commodity"}
    ]
    return random.choice(facts)

# Tool definitions for the AI model
tools = [
    {
        "type": "function",
        "function": {
            "name": "make_coffee",
            "description": "Generate a coffee recipe for the specified type",
            "parameters": {
                "type": "object",
                "properties": {
                    "coffee_type": {
                        "type": "string",
                        "enum": ["espresso", "cappuccino", "latte", "americano"],
                        "description": "Type of coffee to make"
                    }
                },
                "required": ["coffee_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_coffee_fact",
            "description": "Get a random interesting fact about coffee",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]
```

### Example 2: Game Content Generator with Strict Schema

```python
# Define strict output schema
game_content_schema = {
    "type": "object",
    "properties": {
        "character": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 20},
                "class": {"type": "string", "enum": ["warrior", "mage", "rogue", "cleric"]},
                "health": {"type": "integer", "minimum": 50, "maximum": 100},
                "mana": {"type": "integer", "minimum": 0, "maximum": 100}
            },
            "required": ["name", "class", "health"],
            "additionalProperties": False
        },
        "quest_step": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "minLength": 20, "maxLength": 200},
                "choices": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "minLength": 10, "maxLength": 100},
                            "consequence": {"type": "string", "minLength": 10, "maxLength": 100}
                        },
                        "required": ["description"]
                    }
                }
            },
            "required": ["description", "choices"],
            "additionalProperties": False
        }
    },
    "required": ["character", "quest_step"],
    "additionalProperties": False
}
```

## 1.7 Best Practices Summary

### Design Principles
1. **Start simple**: Begin with basic schemas and add complexity gradually
2. **Validate early**: Implement validation at every step
3. **Handle errors gracefully**: Provide fallback mechanisms
4. **Test thoroughly**: Use diverse test cases and edge cases
5. **Document clearly**: Explain schemas and expected behaviors

### Performance Considerations
- **Minimize schema complexity**: Simpler schemas are faster to validate
- **Use appropriate constraints**: Don't over-constrain if not necessary
- **Cache validated schemas**: Reuse validation results when possible
- **Monitor performance**: Track validation time and error rates

### Security Guidelines
- **Validate all inputs**: Never trust user-provided data
- **Use allowlists**: Explicitly allow only necessary functions
- **Implement rate limiting**: Prevent abuse of function calls
- **Log appropriately**: Balance observability with privacy
- **Sanitize outputs**: Clean data before returning to users

## Next Steps

Now that you understand the fundamentals of function calling and structured outputs, proceed to the hands-on exercises in the accompanying Jupyter notebook. You'll implement these concepts with real code and build working examples using open-source tools like HuggingFace and Ollama.

**Upcoming Topics:**
- Prompt engineering techniques
- Model evaluation frameworks
- Local inference deployment
- Performance optimization strategies
