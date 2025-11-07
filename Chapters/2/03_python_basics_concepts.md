# Python Basics - Concepts and Theory

## Introduction

This section covers the fundamental concepts of Python programming, organized into clear theoretical explanations that you can reference while practicing with the accompanying Jupyter notebook. This separation allows you to focus on understanding the concepts first, then apply them through hands-on coding exercises.

## Table of Contents
1. [Variables and Data Types](#variables-and-data-types)
2. [Data Structures](#data-structures)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [Modules and Packages](#modules-and-packages)
6. [Exception Handling](#exception-handling)
7. [File I/O and JSON](#file-io-and-json)
8. [Debugging Concepts](#debugging-concepts)

---

## Variables and Data Types

### Dynamic Typing
Python uses dynamic typing, meaning you don't need to declare variable types explicitly. The interpreter determines the type based on the value assigned.

### Basic Types
- **Integers**: Whole numbers (e.g., `42`, `-17`)
- **Floats**: Decimal numbers (e.g., `3.14159`, `-2.5`)
- **Strings**: Text data (e.g., `"Hello"`, `'Python'`)
- **Booleans**: True/False values (`True`, `False`)
- **None**: Represents absence of value

### Type Conversion
Python provides built-in functions to convert between types:
- `int()`: Convert to integer
- `float()`: Convert to float
- `str()`: Convert to string
- `bool()`: Convert to boolean

### String Formatting
Modern Python uses f-strings for formatting:
```python
name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old."
```

---

## Data Structures

### Lists
- **Purpose**: Ordered, mutable collections
- **Characteristics**: Can contain mixed types, supports indexing and slicing
- **Common Operations**: append, insert, remove, pop, sort, reverse
- **Use Cases**: Sequences of items, stacks, queues

### Dictionaries
- **Purpose**: Key-value mappings
- **Characteristics**: Unordered (before Python 3.7), mutable, keys must be hashable
- **Common Operations**: get, keys, values, items, update, pop
- **Use Cases**: Configuration data, JSON-like structures, lookups

### Tuples
- **Purpose**: Ordered, immutable sequences
- **Characteristics**: Fixed size, hashable (if contents are hashable)
- **Common Operations**: Indexing, unpacking, counting
- **Use Cases**: Function return values, dictionary keys, fixed records

### Sets
- **Purpose**: Unique, unordered collections
- **Characteristics**: Automatic deduplication, supports mathematical set operations
- **Common Operations**: add, remove, union, intersection, difference
- **Use Cases**: Removing duplicates, membership testing, set operations

---

## Control Flow

### Conditional Statements
- **if/elif/else**: Execute different code blocks based on conditions
- **Ternary Operator**: Compact conditional expressions
- **Best Practices**: Clear conditions, proper indentation, meaningful variable names

### Loops
- **for loops**: Iterate over sequences (lists, strings, ranges)
- **while loops**: Continue while condition is true
- **Loop Control**: break (exit loop), continue (skip to next iteration)
- **enumerate()**: Get both index and value when iterating

### List Comprehensions
- **Purpose**: Create lists concisely
- **Syntax**: `[expression for item in iterable if condition]`
- **Benefits**: More readable and often faster than traditional loops

---

## Functions

### Function Definition
- **Purpose**: Reusable blocks of code
- **Components**: Name, parameters, docstring, body, return statement
- **Best Practices**: Single responsibility, clear naming, proper documentation

### Parameters and Arguments
- **Positional Arguments**: Passed by position
- **Keyword Arguments**: Passed by name
- **Default Parameters**: Optional parameters with default values
- **Variable Arguments**: *args (positional), **kwargs (keyword)

### Return Values
- **Single Value**: Return one object
- **Multiple Values**: Return tuple and unpack
- **None**: Implicit return for functions without return statement

### Type Hints (Python 3.5+)
- **Purpose**: Document expected types
- **Syntax**: `def function(param: type) -> return_type:`
- **Benefits**: Better IDE support, documentation, early error detection

---

## Modules and Packages

### Module System
- **Purpose**: Organize code into reusable files
- **Import Methods**: `import module`, `from module import function`, `import module as alias`
- **Standard Library**: Extensive collection of built-in modules

### Creating Modules
- **Structure**: Python file with functions, classes, and variables
- **Documentation**: Module-level docstrings
- **Constants**: UPPERCASE naming convention
- **Private Elements**: Prefix with underscore (_private_function)

### Packages
- **Purpose**: Organize related modules into directories
- **Structure**: Directory with `__init__.py` file
- **Imports**: Use relative imports within packages
- **Metadata**: Version, author, description in `__init__.py`

---

## Exception Handling

### Error Types
- **Syntax Errors**: Code structure problems
- **Runtime Errors**: Occur during execution
- **Logical Errors**: Incorrect program behavior

### Exception Hierarchy
- **BaseException**: Root of all exceptions
- **Exception**: Base for non-system-exiting exceptions
- **Specific Exceptions**: ValueError, TypeError, FileNotFoundError, etc.

### try/except Structure
```
try:
    # Code that might raise an exception
    risky_operation()
except SpecificException as e:
    # Handle specific exception
    handle_error(e)
except Exception as e:
    # Handle any other exception
    handle_unexpected(e)
else:
    # Execute if no exception occurred
    success_handler()
finally:
    # Always execute (cleanup)
    cleanup()
```

### Custom Exceptions
- **Purpose**: Domain-specific error handling
- **Creation**: Inherit from Exception or specific exception classes
- **Usage**: Raise with meaningful messages and context

---

## File I/O and JSON

### File Operations
- **Context Managers**: Use `with` statement for automatic file closing
- **Encoding**: Always specify encoding (usually UTF-8)
- **Modes**: 'r' (read), 'w' (write), 'a' (append), 'b' (binary)
- **Error Handling**: Handle FileNotFoundError, IOError, PermissionError

### Reading Files
- **read()**: Entire file content
- **readline()**: One line at a time
- **readlines()**: All lines as list
- **Iteration**: Process file line by line for large files

### Writing Files
- **write()**: Write string to file
- **writelines()**: Write list of strings
- **Formatting**: Use proper line endings and encoding

### JSON Processing
- **json.dumps()**: Convert Python object to JSON string
- **json.loads()**: Parse JSON string to Python object
- **json.dump()**: Write JSON to file
- **json.load()**: Read JSON from file
- **Validation**: Check data structure and types before processing

---

## Debugging Concepts

### Debugging Strategy
1. **Reproduce the Problem**: Consistently recreate the error
2. **Isolate the Issue**: Narrow down the problematic code
3. **Form Hypotheses**: Make educated guesses about the cause
4. **Test Solutions**: Try fixes and verify results
5. **Document Findings**: Record what you learned

### Print Debugging
- **Purpose**: Quick inspection of variable values and program flow
- **Best Practices**: Strategic placement, clear labels, remove when done
- **Limitations**: Can be messy, affects performance, hard to manage

### Python Debugger (pdb)
- **Purpose**: Interactive debugging with breakpoints
- **Commands**: 
  - `n` (next): Execute next line
  - `s` (step): Step into function calls
  - `c` (continue): Continue until next breakpoint
  - `p` (print): Print variable value
  - `l` (list): Show current code context
  - `q` (quit): Exit debugger

### Logging
- **Purpose**: Professional debugging and monitoring
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Configuration**: Format, level, output destination
- **Best Practices**: Use appropriate levels, include context, structured formatting

### Common Debugging Scenarios
- **Logic Errors**: Program runs but produces wrong results
- **Type Errors**: Incorrect data types or operations
- **Index Errors**: Accessing non-existent list/string indices
- **Key Errors**: Accessing non-existent dictionary keys
- **File Errors**: Missing files, permission issues, encoding problems

---

## Best Practices Summary

### Code Organization
- Use meaningful variable and function names
- Keep functions focused on single responsibility
- Document with docstrings and comments
- Follow PEP 8 style guidelines

### Error Handling
- Handle specific exceptions when possible
- Provide meaningful error messages
- Clean up resources in finally blocks
- Log errors for debugging and monitoring

### File Operations
- Always use context managers (with statement)
- Specify encoding explicitly
- Validate file existence and permissions
- Handle different file formats appropriately

### Module Design
- Create focused, reusable modules
- Use clear import statements
- Document module purpose and usage
- Follow naming conventions

---

## Next Steps

Now that you understand the concepts, practice them with the accompanying Jupyter notebook: `02_python_basics_exercises.ipynb`

The notebook contains:
- Interactive coding exercises for each concept
- Real-world examples and scenarios
- Debugging practice problems
- File processing challenges
- Step-by-step solutions

## Additional Resources

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Python Module Index](https://docs.python.org/3/py-modindex.html)
- [Real Python Articles](https://realpython.com/)

**Remember: Understanding concepts is the first step. Practice is essential for mastery!**
