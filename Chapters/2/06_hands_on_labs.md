# Part 6: Hands-on Labs

## Introduction

This section contains comprehensive hands-on labs that combine all the concepts learned in previous sections. Each lab is a complete project that you can work through to gain practical experience with Python programming and environment management for AI engineering.

## Table of Contents
1. [Lab 1: JSON Data Processing Project](#lab-1-json-data-processing-project)
2. [Lab 2: Environment Setup for AI Development](#lab-2-environment-setup-for-ai-development)
3. [Lab 3: Debugging and Error Handling](#lab-3-debugging-and-error-handling)
4. [Lab 4: Reproducible Research Project](#lab-4-reproducible-research-project)
5. [Lab 5: Advanced AI Environment](#lab-5-advanced-ai-environment)

---

## Lab 1: JSON Data Processing Project

### Objective
Create a complete data processing pipeline that reads JSON data, performs analysis, and generates visualizations.

### Project Structure
```
json-data-project/
├── data/
│   ├── students.json
│   └── courses.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── analyzer.py
│   └── visualizer.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
├── environment.yml
└── README.md
```

### Step 1: Create Sample Data

Create `data/students.json`:
```json
[
  {
    "id": 1,
    "name": "Alice Johnson",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8,
    "courses": ["CS101", "CS102", "MATH201"],
    "grades": {"CS101": 92, "CS102": 88, "MATH201": 85}
  },
  {
    "id": 2,
    "name": "Bob Smith",
    "age": 22,
    "major": "Data Science",
    "gpa": 3.6,
    "courses": ["DS101", "DS102", "STAT201"],
    "grades": {"DS101": 89, "DS102": 91, "STAT201": 78}
  },
  {
    "id": 3,
    "name": "Carol Davis",
    "age": 21,
    "major": "Computer Science",
    "gpa": 3.9,
    "courses": ["CS101", "CS103", "MATH202"],
    "grades": {"CS101": 95, "CS103": 92, "MATH202": 88}
  }
]
```

Create `data/courses.json`:
```json
[
  {
    "code": "CS101",
    "name": "Introduction to Programming",
    "credits": 3,
    "department": "Computer Science"
  },
  {
    "code": "CS102",
    "name": "Data Structures",
    "credits": 3,
    "department": "Computer Science"
  },
  {
    "code": "DS101",
    "name": "Data Science Fundamentals",
    "credits": 3,
    "department": "Data Science"
  }
]
```

### Step 2: Create Data Loader Module

Create `src/data_loader.py`:
```python
import json
from pathlib import Path
from typing import List, Dict, Any

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_students(self) -> List[Dict[str, Any]]:
        """Load student data from JSON file."""
        try:
            with open(self.data_dir / "students.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("students.json file not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in students.json: {e}")
    
    def load_courses(self) -> List[Dict[str, Any]]:
        """Load course data from JSON file."""
        try:
            with open(self.data_dir / "courses.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("courses.json file not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in courses.json: {e}")
    
    def validate_student_data(self, students: List[Dict[str, Any]]) -> bool:
        """Validate student data structure."""
        required_fields = ['id', 'name', 'age', 'major', 'gpa', 'courses', 'grades']
        
        for student in students:
            for field in required_fields:
                if field not in student:
                    raise ValueError(f"Missing required field '{field}' in student data")
            
            if not isinstance(student['age'], int) or student['age'] < 0:
                raise ValueError("Student age must be a positive integer")
            
            if not isinstance(student['gpa'], (int, float)) or not (0 <= student['gpa'] <= 4):
                raise ValueError("Student GPA must be between 0 and 4")
        
        return True
```

### Step 3: Create Analyzer Module

Create `src/analyzer.py`:
```python
import statistics
from typing import List, Dict, Any

class StudentAnalyzer:
    def __init__(self, students: List[Dict[str, Any]], courses: List[Dict[str, Any]]):
        self.students = students
        self.courses = courses
        self.course_map = {course['code']: course for course in courses}
    
    def calculate_class_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for all students."""
        if not self.students:
            return {"error": "No student data available"}
        
        ages = [student['age'] for student in self.students]
        gpas = [student['gpa'] for student in self.students]
        
        # Calculate grade statistics for each course
        course_grades = {}
        for student in self.students:
            for course_code, grade in student['grades'].items():
                if course_code not in course_grades:
                    course_grades[course_code] = []
                course_grades[course_code].append(grade)
        
        course_stats = {}
        for course_code, grades in course_grades.items():
            course_stats[course_code] = {
                "mean": statistics.mean(grades),
                "median": statistics.median(grades),
                "std_dev": statistics.stdev(grades) if len(grades) > 1 else 0,
                "count": len(grades)
            }
        
        return {
            "total_students": len(self.students),
            "age_statistics": {
                "mean": statistics.mean(ages),
                "median": statistics.median(ages),
                "min": min(ages),
                "max": max(ages)
            },
            "gpa_statistics": {
                "mean": statistics.mean(gpas),
                "median": statistics.median(gpas),
                "min": min(gpas),
                "max": max(gpas)
            },
            "course_statistics": course_stats
        }
    
    def find_top_performers(self, threshold: float = 90) -> List[Dict[str, Any]]:
        """Find students with average grades above threshold."""
        top_performers = []
        
        for student in self.students:
            grades = list(student['grades'].values())
            avg_grade = statistics.mean(grades)
            
            if avg_grade >= threshold:
                top_performers.append({
                    "name": student['name'],
                    "major": student['major'],
                    "average_grade": avg_grade,
                    "gpa": student['gpa']
                })
        
        return sorted(top_performers, key=lambda x: x['average_grade'], reverse=True)
    
    def analyze_by_major(self) -> Dict[str, Any]:
        """Analyze students grouped by major."""
        major_groups = {}
        
        for student in self.students:
            major = student['major']
            if major not in major_groups:
                major_groups[major] = []
            major_groups[major].append(student)
        
        major_stats = {}
        for major, students in major_groups.items():
            gpas = [s['gpa'] for s in students]
            grades = []
            for student in students:
                grades.extend(student['grades'].values())
            
            major_stats[major] = {
                "student_count": len(students),
                "avg_gpa": statistics.mean(gpas),
                "avg_grade": statistics.mean(grades) if grades else 0,
                "students": [{"name": s['name'], "gpa": s['gpa']} for s in students]
            }
        
        return major_stats
```

### Step 4: Create Visualizer Module

Create `src/visualizer.py`:
```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

class DataVisualizer:
    def __init__(self, style: str = "seaborn"):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_age_distribution(self, ages: List[int]) -> None:
        """Create histogram of student ages."""
        plt.figure(figsize=(10, 6))
        plt.hist(ages, bins=10, edgecolor='black', alpha=0.7)
        plt.xlabel('Age')
        plt.ylabel('Number of Students')
        plt.title('Student Age Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gpa_vs_age(self, students: List[Dict[str, Any]]) -> None:
        """Create scatter plot of GPA vs Age."""
        ages = [s['age'] for s in students]
        gpas = [s['gpa'] for s in students]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(ages, gpas, alpha=0.7, s=100)
        plt.xlabel('Age')
        plt.ylabel('GPA')
        plt.title('GPA vs Age')
        plt.grid(True, alpha=0.3)
        plt.savefig('gpa_vs_age.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_course_performance(self, course_stats: Dict[str, Any]) -> None:
        """Create bar chart of course performance."""
        courses = list(course_stats.keys())
        means = [course_stats[course]['mean'] for course in courses]
        
        plt.figure
