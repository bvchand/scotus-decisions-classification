from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

students = {
    1: {
        'name': 'bharathi',
        'age': 12,
        'year': 'aug 20'
    }
}

class Student(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    year: Optional[str] = None


@app.get("/")  # '/' means 'home page'
def home():
    return {"name": "SCOTUS decisions"}


@app.get("/get-student/{student_id}")  # path parameter        http://127.0.0.1:8000/get-student/1
def get_student(student_id: int = Path(None, description="give the id", gt=0)):
    return students[student_id]


@app.get("/get-by-name/{student_id}")  # name - query parameter       http://127.0.0.1:8000/get-by-name?name=bharathi
def get_student(*, student_id: int, name: Optional[str] = None, test: int):
    for student_id in students:
        if students[student_id]["name"] == name:    return students[student_id]
    return {"Data": "Not found"}


@app.post("/create-student/{student_id}")
def create_student(student_id: int, student: Student):
    if student_id in students:  return {"Error": "Student exists"}
    students[student_id] = student
    return students[student_id]


@app.put("/update-student/{student_id}")
def update_student(student_id: int, student: Student):
    if student_id not in students:
        return {"Error": "Not found"}
    if student.name != None:    students[student_id].name = student.name
    if student.age != None:    students[student_id].age = student.age
    if student.year != None:    students[student_id].year = student.year
    return students[student_id]
