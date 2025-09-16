#!/usr/bin/env python3
"""
Professional University Timetable Generator with Intelligent Teacher-Subject Assignment
Features: Expertise-based assignment, workload balancing, constraint optimization
"""

import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
import itertools
from collections import defaultdict
import pandas as pd
import math

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

@dataclass
class Room:
    id: str
    capacity: int
    room_type: str
    building: str
    floor: int
    has_projector: bool = True
    has_computer: bool = False

@dataclass
class Teacher:
    id: str
    name: str
    department: str
    expertise_areas: List[str] = field(default_factory=list)
    specialization_level: str = "Advanced"  # Beginner, Advanced, Expert
    subjects: List[str] = field(default_factory=list)
    max_workload: int = 20
    current_workload: int = 0
    preferred_class_types: List[str] = field(default_factory=list)
    expertise_scores: Dict[str, float] = field(default_factory=dict)
    preferred_times: List[Tuple[int, int]] = field(default_factory=list)
    unavailable_times: List[Tuple[int, int]] = field(default_factory=list)
    max_hours_per_day: int = 6

@dataclass
class Subject:
    id: str
    name: str
    credits: int
    hours_per_week: int
    subject_type: str
    difficulty: str = "Intermediate"  # Beginner, Intermediate, Advanced, Expert
    requires_computer: bool = False
    requires_projector: bool = True
    prerequisite: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class Division:
    id: str
    name: str
    strength: int
    semester: int
    subjects: List[str]

@dataclass
class TimeSlot:
    day: int
    period: int

    def __str__(self):
        return f"Day {self.day} Period {self.period}"

@dataclass
class ScheduleEntry:
    division_id: str
    subject_id: str
    teacher_id: str
    room_id: str
    time_slot: TimeSlot
    assignment_score: float = 0.0

class TeacherSubjectAssigner:
    """Professional-grade teacher-subject assignment system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.assignment_weights = config['assignment_algorithm']['scoring_weights']
        self.constraints = config['assignment_algorithm']['constraints']
    
    def calculate_expertise_score(self, teacher: Teacher, subject: Subject) -> float:
        """Calculate how well teacher's expertise matches subject requirements"""
        # Get subject expertise requirements from config
        subject_info = self._get_subject_info(subject.id)
        if not subject_info:
            return 0.5
        
        # Base score from specialization level match
        level_scores = {"Beginner": 0.3, "Intermediate": 0.5, "Advanced": 0.7, "Expert": 1.0}
        required_level = subject_info.get('difficulty', 'Intermediate')
        teacher_level = teacher.specialization_level
        
        level_score = level_scores.get(teacher_level, 0.5)
        required_score = level_scores.get(required_level, 0.5)
        
        # Penalty if teacher level is below required level
        if level_score < required_score:
            level_match = level_score / required_score * 0.6
        else:
            level_match = min(1.0, level_score / required_score)
        
        # Expertise area match
        subject_areas = self._extract_subject_areas(subject)
        if not subject_areas:
            expertise_match = 0.5
        else:
            matches = len(set(teacher.expertise_areas) & set(subject_areas))
            expertise_match = matches / len(subject_areas) if subject_areas else 0.5
        
        # Combined score
        return (level_match * 0.6) + (expertise_match * 0.4)
    
    def calculate_preference_score(self, teacher: Teacher, subject: Subject) -> float:
        """Calculate teacher's preference for the subject"""
        dept_config = self.config['faculty_config']['departments'].get(teacher.department, {})
        teacher_config = dept_config.get('teacher_expertise', {}).get(teacher.name, {})
        
        preferences = teacher_config.get('subject_preferences', [])
        if subject.id in preferences:
            # Higher score for higher preference order
            index = preferences.index(subject.id)
            return max(0.2, 1.0 - (index * 0.15))
        return 0.1
    
    def calculate_workload_score(self, teacher: Teacher) -> float:
        """Calculate workload balance score (higher score for less loaded teachers)"""
        if teacher.max_workload == 0:
            return 0.0
        utilization = teacher.current_workload / teacher.max_workload
        return max(0.0, 1.0 - utilization)
    
    def calculate_class_type_score(self, teacher: Teacher, subject: Subject) -> float:
        """Calculate how well subject type matches teacher's preferences"""
        if subject.subject_type in teacher.preferred_class_types:
            return 1.0
        elif len(teacher.preferred_class_types) == 0:
            return 0.5
        else:
            return 0.2
    
    def calculate_assignment_score(self, teacher: Teacher, subject: Subject) -> float:
        """Calculate overall assignment score using weighted factors"""
        expertise_score = self.calculate_expertise_score(teacher, subject)
        preference_score = self.calculate_preference_score(teacher, subject)
        workload_score = self.calculate_workload_score(teacher)
        class_type_score = self.calculate_class_type_score(teacher, subject)
        
        # Specialization level bonus
        level_scores = {"Beginner": 0.6, "Advanced": 0.8, "Expert": 1.0}
        specialization_score = level_scores.get(teacher.specialization_level, 0.7)
        
        # Weighted combination
        weights = self.assignment_weights
        total_score = (
            expertise_score * weights['expertise_match'] +
            specialization_score * weights['specialization_level'] +
            workload_score * weights['workload_balance'] +
            preference_score * weights['preference_match'] +
            class_type_score * weights['class_type_preference']
        ) / 100.0
        
        return min(1.0, max(0.0, total_score))
    
    def assign_subjects_to_teachers(self, teachers: List[Teacher], subjects: List[Subject]) -> Dict[str, List[str]]:
        """Professional algorithm for optimal teacher-subject assignment"""
        print("🎯 Starting intelligent teacher-subject assignment...")
        
        # Reset teacher assignments
        for teacher in teachers:
            teacher.subjects = []
            teacher.current_workload = 0
            teacher.expertise_scores = {}
        
        # Calculate all possible assignments with scores
        assignment_candidates = []
        
        for teacher in teachers:
            for subject in subjects:
                # Check if teacher is from relevant department
                if self._teacher_can_teach_subject(teacher, subject):
                    score = self.calculate_assignment_score(teacher, subject)
                    teacher.expertise_scores[subject.id] = score
                    
                    if score >= self.constraints['min_expertise_score']:
                        assignment_candidates.append((teacher.id, subject.id, score))
        
        # Sort by score (highest first)
        assignment_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Track assignments
        subject_assignments = {subject.id: [] for subject in subjects}
        teacher_workloads = {teacher.id: 0 for teacher in teachers}
        
        # First pass: High-quality assignments
        for teacher_id, subject_id, score in assignment_candidates:
            teacher = next(t for t in teachers if t.id == teacher_id)
            subject = next(s for s in subjects if s.id == subject_id)
            
            if (teacher_workloads[teacher_id] < teacher.max_workload and
                len(subject_assignments[subject_id]) < 2):  # Max 2 teachers per subject
                
                teacher.subjects.append(subject_id)
                subject_assignments[subject_id].append(teacher_id)
                teacher_workloads[teacher_id] += subject.hours_per_week
                teacher.current_workload += subject.hours_per_week
        
        # Second pass: Ensure all subjects are covered
        uncovered_subjects = [sid for sid, teachers_list in subject_assignments.items() if not teachers_list]
        
        for subject_id in uncovered_subjects:
            subject = next(s for s in subjects if s.id == subject_id)
            best_teacher = None
            best_score = -1
            
            for teacher in teachers:
                if (self._teacher_can_teach_subject(teacher, subject) and
                    teacher_workloads[teacher.id] < teacher.max_workload):
                    
                    score = teacher.expertise_scores.get(subject_id, 0)
                    if score > best_score:
                        best_score = score
                        best_teacher = teacher
            
            if best_teacher:
                best_teacher.subjects.append(subject_id)
                subject_assignments[subject_id].append(best_teacher.id)
                teacher_workloads[best_teacher.id] += subject.hours_per_week
                best_teacher.current_workload += subject.hours_per_week
        
        # Generate assignment report
        self._print_assignment_report(teachers, subjects, subject_assignments)
        
        return subject_assignments
    
    def _teacher_can_teach_subject(self, teacher: Teacher, subject: Subject) -> bool:
        """Check if teacher is qualified to teach the subject"""
        # Check department match
        dept_config = self.config['faculty_config']['departments'].get(teacher.department, {})
        subject_prefixes = dept_config.get('subject_prefixes', [])
        
        return any(subject.id.startswith(prefix) for prefix in subject_prefixes)
    
    def _get_subject_info(self, subject_id: str) -> Optional[Dict]:
        """Get subject information from config"""
        for dept_subjects in self.config['subject_templates'].values():
            for semester_subjects in dept_subjects['semesters'].values():
                for subject_info in semester_subjects:
                    if subject_info['code'] == subject_id:
                        return subject_info
        return None
    
    def _extract_subject_areas(self, subject: Subject) -> List[str]:
        """Extract relevant expertise areas for a subject"""
        subject_mappings = {
            'CS101': ['Programming', 'Algorithms'],
            'CS102': ['Programming', 'Lab Instruction'],
            'CS201': ['Data Structures', 'Algorithms'],
            'CS202': ['Data Structures', 'Lab Instruction'],
            'CS301': ['Database Systems', 'Software Engineering'],
            'CS302': ['Database Systems', 'Lab Instruction'],
            'CS303': ['Operating Systems', 'System Programming'],
            'CS402': ['Machine Learning', 'Artificial Intelligence'],
            'MA101': ['Calculus', 'Mathematical Analysis'],
            'MA202': ['Probability', 'Statistics']
        }
        return subject_mappings.get(subject.id, [])
    
    def _print_assignment_report(self, teachers: List[Teacher], subjects: List[Subject], assignments: Dict[str, List[str]]):
        """Print detailed assignment report"""
        print("\n📊 TEACHER-SUBJECT ASSIGNMENT REPORT")
        print("=" * 60)
        
        for teacher in teachers:
            if teacher.subjects:
                avg_score = sum(teacher.expertise_scores.get(subj, 0) for subj in teacher.subjects) / len(teacher.subjects)
                utilization = teacher.current_workload / teacher.max_workload * 100
                
                print(f"\n👨‍🏫 {teacher.name} ({teacher.department})")
                print(f"   📈 Expertise Level: {teacher.specialization_level}")
                print(f"   📚 Subjects ({len(teacher.subjects)}): {', '.join(teacher.subjects)}")
                print(f"   📊 Avg Expertise Score: {avg_score:.2f}")
                print(f"   ⚖️ Workload: {teacher.current_workload}/{teacher.max_workload} hours ({utilization:.1f}%)")
        
        # Coverage report
        uncovered = [sid for sid, teacher_list in assignments.items() if not teacher_list]
        if uncovered:
            print(f"\n⚠️ Uncovered Subjects: {uncovered}")
        else:
            print(f"\n✅ All {len(subjects)} subjects successfully assigned!")

class ProTimetableGenerator:
    def __init__(self, config_file: str = "pro_timetable_config.json"):
        """Initialize professional timetable generator"""
        self.config = self.load_config(config_file)
        
        # Initialize data structures
        self.rooms: List[Room] = []
        self.teachers: List[Teacher] = []
        self.subjects: List[Subject] = []
        self.divisions: List[Division] = []
        self.schedule: List[ScheduleEntry] = []
        
        # Assignment system
        self.assigner = TeacherSubjectAssigner(self.config)
        
        # Get configuration values
        self.days = self.config['general_settings']['days_per_week']
        self.periods_per_day = self.config['general_settings']['periods_per_day']
        
        # Constraint tracking
        self.room_schedule = {}
        self.teacher_schedule = {}
        self.division_schedule = {}

    def load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"✅ Professional config loaded from {config_file}")
                return config
        except FileNotFoundError:
            print(f"❌ Config file {config_file} not found!")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON config: {e}")
            raise

    def safe_random_int(self, min_val: int, max_val: int) -> int:
        """Safely generate random integer"""
        if min_val == max_val:
            return min_val
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        return random.randint(min_val, max_val)

    def generate_dummy_data(self):
        """Generate comprehensive dummy data with professional teacher assignments"""
        print("🏗️ Generating professional-grade dummy data...")
        
        self._generate_rooms()
        self._generate_subjects()
        self._generate_teachers_with_expertise()
        self._generate_divisions()
        
        # Professional teacher-subject assignment
        self.assigner.assign_subjects_to_teachers(self.teachers, self.subjects)
        
        total_students = sum(div.strength for div in self.divisions)
        print(f"✅ Generated: {len(self.rooms)} rooms, {len(self.teachers)} teachers, "
              f"{len(self.subjects)} subjects, {len(self.divisions)} divisions "
              f"({total_students} total students)")

    def _generate_rooms(self):
        """Generate rooms based on configuration"""
        building_config = self.config['building_config']
        room_types = self.config['room_types']
        room_id = 1

        for building in building_config['buildings']:
            for floor in range(1, building_config['floors_per_building'] + 1):
                for room_type, type_config in room_types.items():
                    for i in range(building_config['rooms_per_floor_per_type']):
                        capacity_variance = self.safe_random_int(
                            -type_config['capacity_variance'],
                            type_config['capacity_variance']
                        )

                        self.rooms.append(Room(
                            id=f"{building}{floor}{room_id:02d}",
                            capacity=type_config['base_capacity'] + capacity_variance,
                            room_type=room_type,
                            building=building,
                            floor=floor,
                            has_computer=type_config['has_computer'],
                            has_projector=type_config['has_projector']
                        ))
                        room_id += 1

    def _generate_subjects(self):
        """Generate subjects with enhanced metadata"""
        subject_templates = self.config['subject_templates']
        
        for dept_name, dept_config in subject_templates.items():
            for semester, subjects in dept_config['semesters'].items():
                for subj_data in subjects:
                    self.subjects.append(Subject(
                        id=subj_data['code'],
                        name=subj_data['name'],
                        credits=subj_data['credits'],
                        hours_per_week=subj_data['hours'],
                        subject_type=subj_data['type'],
                        difficulty=subj_data.get('difficulty', 'Intermediate'),
                        requires_computer=subj_data.get('requires_computer', False),
                        requires_projector=subj_data.get('requires_projector', True),
                        prerequisite=subj_data.get('prerequisite'),
                        prerequisites=subj_data.get('prerequisites', [])
                    ))

    def _generate_teachers_with_expertise(self):
        """Generate teachers with professional expertise profiles"""
        faculty_config = self.config['faculty_config']
        teacher_id = 1

        for dept_name, dept_config in faculty_config['departments'].items():
            teacher_expertise = dept_config.get('teacher_expertise', {})
            
            for teacher_name, expertise_data in teacher_expertise.items():
                # Generate time preferences
                pref_config = self.config['teacher_preferences']
                
                preferred_times = []
                pref_range = pref_config['preferred_slots_per_teacher']
                pref_count = self.safe_random_int(pref_range[0], pref_range[1])
                
                for _ in range(pref_count):
                    preferred_times.append((
                        self.safe_random_int(0, self.days-1),
                        self.safe_random_int(0, self.periods_per_day-1)
                    ))

                unavailable_times = []
                unavail_range = pref_config['unavailable_slots_per_teacher']
                unavail_count = self.safe_random_int(unavail_range[0], unavail_range[1])
                
                for _ in range(unavail_count):
                    unavailable_times.append((
                        self.safe_random_int(0, self.days-1),
                        self.safe_random_int(0, self.periods_per_day-1)
                    ))

                hours_range = pref_config['max_hours_per_day_range']
                max_hours = self.safe_random_int(hours_range[0], hours_range[1])

                self.teachers.append(Teacher(
                    id=f"T{teacher_id:03d}",
                    name=teacher_name,
                    department=dept_name,
                    expertise_areas=expertise_data.get('expertise_areas', []),
                    specialization_level=expertise_data.get('specialization_level', 'Advanced'),
                    max_workload=expertise_data.get('max_workload', 20),
                    preferred_class_types=expertise_data.get('preferred_class_types', []),
                    preferred_times=preferred_times,
                    unavailable_times=unavailable_times,
                    max_hours_per_day=max_hours
                ))
                teacher_id += 1

            

    def _generate_divisions(self):
        """Generate divisions based on configuration"""
        division_config = self.config['division_config']
        subject_templates = self.config['subject_templates']
        target_students = self.config['general_settings']['target_total_students']
        
        total_students = 0
        division_id = 1

        # Create semester-wise subject mapping
        semester_subjects = {}
        for dept_name, dept_config in subject_templates.items():
            for semester, subjects in dept_config['semesters'].items():
                sem_num = int(semester)
                if sem_num not in semester_subjects:
                    semester_subjects[sem_num] = []
                semester_subjects[sem_num].extend([s['code'] for s in subjects])

        # Generate divisions
        for semester, div_count in division_config['divisions_per_semester'].items():
            sem_num = int(semester)
            for div_num in range(div_count):
                strength_range = division_config['strength_range']
                strength = self.safe_random_int(strength_range['min'], strength_range['max'])

                # Select subjects for this division
                available_subjects = semester_subjects.get(sem_num, [])
                if available_subjects:
                    subjects_range = division_config['subjects_per_division']
                    num_subjects = self.safe_random_int(
                        subjects_range['min'],
                        min(subjects_range['max'], len(available_subjects))
                    )

                    sample_size = min(num_subjects, len(available_subjects))
                    if sample_size > 0:
                        div_subjects = random.sample(available_subjects, sample_size)
                    else:
                        div_subjects = []
                else:
                    div_subjects = []

                self.divisions.append(Division(
                    id=f"DIV{division_id:03d}",
                    name=f"Sem-{sem_num} Div-{chr(65+div_num)}",
                    strength=strength,
                    semester=sem_num,
                    subjects=div_subjects
                ))
                total_students += strength
                division_id += 1

                if total_students >= target_students:
                    return

    def check_hard_constraints(self, division_id: str, subject_id: str,
                             teacher_id: str, room_id: str, time_slot: TimeSlot) -> bool:
        """Enhanced hard constraint checking"""
        # Get objects
        division = next((d for d in self.divisions if d.id == division_id), None)
        subject = next((s for s in self.subjects if s.id == subject_id), None)
        teacher = next((t for t in self.teachers if t.id == teacher_id), None)
        room = next((r for r in self.rooms if r.id == room_id), None)

        if not all([division, subject, teacher, room]):
            return False

        # Standard hard constraints
        key = (time_slot.day, time_slot.period, room_id)
        if key in self.room_schedule:
            return False

        key = (time_slot.day, time_slot.period, teacher_id)
        if key in self.teacher_schedule:
            return False

        key = (time_slot.day, time_slot.period, division_id)
        if key in self.division_schedule:
            return False

        if division.strength > room.capacity:
            return False

        if subject.subject_type != room.room_type and not (
            subject.subject_type == 'tutorial' and room.room_type == 'lecture'
        ):
            return False

        if subject.requires_computer and not room.has_computer:
            return False

        # Enhanced teacher qualification check
        if subject_id not in teacher.subjects:
            return False

        # Check minimum expertise score
        min_score = self.config['assignment_algorithm']['constraints']['min_expertise_score']
        if teacher.expertise_scores.get(subject_id, 0) < min_score:
            return False

        if (time_slot.day, time_slot.period) in teacher.unavailable_times:
            return False

        daily_hours = sum(1 for entry in self.schedule
                         if entry.teacher_id == teacher_id and
                         entry.time_slot.day == time_slot.day)
        if daily_hours >= teacher.max_hours_per_day:
            return False

        return True

    def calculate_soft_constraint_score(self, division_id: str, subject_id: str,
                                       teacher_id: str, room_id: str, time_slot: TimeSlot) -> float:
        """Enhanced soft constraint scoring"""
        score = 0.0
        weights = self.config['constraint_weights']['soft_constraints']

        teacher = next(t for t in self.teachers if t.id == teacher_id)
        division = next(d for d in self.divisions if d.id == division_id)
        room = next(r for r in self.rooms if r.id == room_id)
        subject = next(s for s in self.subjects if s.id == subject_id)

        # Teacher expertise bonus
        expertise_score = teacher.expertise_scores.get(subject_id, 0)
        score += expertise_score * 15

        # Teacher time preferences
        if (time_slot.day, time_slot.period) in teacher.preferred_times:
            score += weights['teacher_preferences']

        # Daily subject distribution
        daily_subjects = sum(1 for entry in self.schedule
                           if entry.division_id == division_id and
                           entry.time_slot.day == time_slot.day)
        
        if daily_subjects < 3:
            score += weights['daily_subject_limit']
        elif daily_subjects >= 4:
            score -= weights['daily_subject_limit'] * 0.7

        # Avoid lunch hours
        lunch_periods = self.config['time_slot_config']['lunch_periods']
        if time_slot.period not in lunch_periods:
            score += weights['avoid_lunch_hour']

        # Room utilization
        utilization = division.strength / room.capacity if room.capacity > 0 else 0
        if 0.7 <= utilization <= 0.9:
            score += weights['room_utilization']
        elif utilization < 0.5:
            score -= weights['room_utilization'] * 0.6

        # Sequential classes in nearby rooms
        prev_slot = TimeSlot(time_slot.day, time_slot.period - 1) if time_slot.period > 0 else None
        next_slot = TimeSlot(time_slot.day, time_slot.period + 1) if time_slot.period < self.periods_per_day - 1 else None

        for slot in [prev_slot, next_slot]:
            if slot:
                key = (slot.day, slot.period, division_id)
                if key in self.division_schedule:
                    other_room_id = self.division_schedule[key].room_id
                    other_room = next(r for r in self.rooms if r.id == other_room_id)
                    if other_room.building == room.building:
                        score += weights['sequential_rooms']

        # Prerequisite ordering
        if subject.prerequisite:
            prereq_scheduled = any(
                entry.division_id == division_id and
                entry.subject_id == subject.prerequisite and
                (entry.time_slot.day < time_slot.day or
                 (entry.time_slot.day == time_slot.day and entry.time_slot.period < time_slot.period))
                for entry in self.schedule
            )

            if prereq_scheduled:
                score += weights['prerequisite_ordering']
            else:
                score -= weights['prerequisite_ordering'] * 0.6

        return score

    def generate_timetable(self):
        """Generate optimized timetable with enhanced algorithm"""
        print("🚀 Generating professional timetable...")

        # Create assignments list
        assignments = []
        for division in self.divisions:
            for subject_id in division.subjects:
                subject = next(s for s in self.subjects if s.id == subject_id)
                for _ in range(subject.hours_per_week):
                    assignments.append((division.id, subject_id))

        random.shuffle(assignments)
        scheduled = 0
        total_assignments = len(assignments)
        optimization_config = self.config['optimization_settings']

        for division_id, subject_id in assignments:
            best_assignment = None
            best_score = -float('inf')

            subject = next(s for s in self.subjects if s.id == subject_id)
            
            # Get qualified teachers (those who can teach this subject)
            eligible_teachers = [t for t in self.teachers if subject_id in t.subjects]
            
            eligible_rooms = [r for r in self.rooms
                            if (r.room_type == subject.subject_type or
                                (subject.subject_type == 'tutorial' and r.room_type == 'lecture')) and
                            (not subject.requires_computer or r.has_computer)]

            if not eligible_teachers or not eligible_rooms:
                continue

            # Sample for efficiency with larger sample sizes for better quality
            teacher_sample_size = min(len(eligible_teachers), optimization_config['teacher_sample_size'])
            room_sample_size = min(len(eligible_rooms), optimization_config['room_sample_size'])
            
            time_slots = [TimeSlot(d, p) for d in range(self.days) for p in range(self.periods_per_day)]
            time_sample_size = min(len(time_slots), optimization_config['time_slot_sample_size'])

            teacher_sample = random.sample(eligible_teachers, teacher_sample_size) if teacher_sample_size > 0 else []
            room_sample = random.sample(eligible_rooms, room_sample_size) if room_sample_size > 0 else []
            time_sample = random.sample(time_slots, time_sample_size) if time_sample_size > 0 else []

            for teacher in teacher_sample:
                for room in room_sample:
                    for time_slot in time_sample:
                        if self.check_hard_constraints(division_id, subject_id,
                                                     teacher.id, room.id, time_slot):
                            soft_score = self.calculate_soft_constraint_score(
                                division_id, subject_id, teacher.id, room.id, time_slot)
                            
                            # Add expertise score bonus
                            expertise_bonus = teacher.expertise_scores.get(subject_id, 0) * 20
                            total_score = soft_score + expertise_bonus

                            if total_score > best_score:
                                best_score = total_score
                                best_assignment = ScheduleEntry(
                                    division_id=division_id,
                                    subject_id=subject_id,
                                    teacher_id=teacher.id,
                                    room_id=room.id,
                                    time_slot=time_slot,
                                    assignment_score=total_score
                                )

            if best_assignment:
                self.schedule.append(best_assignment)
                
                # Update tracking
                ts = best_assignment.time_slot
                self.room_schedule[(ts.day, ts.period, best_assignment.room_id)] = best_assignment
                self.teacher_schedule[(ts.day, ts.period, best_assignment.teacher_id)] = best_assignment
                self.division_schedule[(ts.day, ts.period, best_assignment.division_id)] = best_assignment

                scheduled += 1

            if scheduled % 50 == 0:
                print(f"📈 Progress: {scheduled}/{total_assignments} ({scheduled/total_assignments*100:.1f}%)")

        success_rate = (scheduled / total_assignments * 100) if total_assignments > 0 else 0
        print(f"🎉 Professional timetable generation complete! Success rate: {success_rate:.1f}%")
        
        target_rate = self.config['general_settings'].get('min_success_rate_target', 90)
        if success_rate >= target_rate:
            print(f"✅ Excellent! Success rate meets professional target ({target_rate}%)")
        else:
            print(f"⚠️ Success rate below professional target ({target_rate}%). Consider adjusting constraints.")

    def print_detailed_statistics(self):
        """Print comprehensive professional statistics"""
        print("\n" + "="*80)
        print("📊 PROFESSIONAL TIMETABLE GENERATION STATISTICS")
        print("="*80)

        total_possible = sum(
            sum(s.hours_per_week for s in self.subjects if s.id in div.subjects)
            for div in self.divisions
        )

        scheduled = len(self.schedule)
        print(f"🎯 Scheduling Performance:")
        print(f"   Total classes to schedule: {total_possible}")
        print(f"   Successfully scheduled: {scheduled}")
        print(f"   Success rate: {scheduled/total_possible*100:.2f}%" if total_possible > 0 else "   Success rate: 0.00%")

        # Teacher expertise analysis
        print(f"\n👨‍🏫 Teacher Assignment Quality:")
        if self.schedule:
            avg_expertise = sum(entry.assignment_score for entry in self.schedule) / len(self.schedule)
            high_quality = sum(1 for entry in self.schedule if entry.assignment_score > 0.7)
            print(f"   Average assignment quality: {avg_expertise:.2f}/100")
            print(f"   High-quality assignments: {high_quality}/{len(self.schedule)} ({high_quality/len(self.schedule)*100:.1f}%)")

        # Workload distribution
        print(f"\n⚖️ Teacher Workload Balance:")
        if self.teachers:
            workloads = [t.current_workload for t in self.teachers if t.current_workload > 0]
            if workloads:
                avg_workload = sum(workloads) / len(workloads)
                min_workload = min(workloads)
                max_workload = max(workloads)
                print(f"   Average workload: {avg_workload:.1f} hours")
                print(f"   Workload range: {min_workload} - {max_workload} hours")
                
                # Balance score
                variance = sum((w - avg_workload) ** 2 for w in workloads) / len(workloads)
                balance_score = max(0, 100 - (variance / avg_workload * 100))
                print(f"   Workload balance score: {balance_score:.1f}/100")

        # Room utilization
        room_usage = defaultdict(int)
        for entry in self.schedule:
            room_usage[entry.room_id] += 1

        if room_usage:
            print(f"\n🏢 Resource Utilization:")
            avg_room_usage = sum(room_usage.values())/len(room_usage) if room_usage else 0
            print(f"   Average classes per room: {avg_room_usage:.1f}")
            
            if room_usage:
                most_used = max(room_usage.items(), key=lambda x: x[1])
                least_used = min(room_usage.items(), key=lambda x: x[1])
                print(f"   Most utilized room: {most_used[0]} ({most_used[1]} classes)")
                print(f"   Least utilized room: {least_used[0]} ({least_used[1]} classes)")

        # Daily distribution
        daily_dist = defaultdict(int)
        for entry in self.schedule:
            daily_dist[entry.time_slot.day] += 1

        if daily_dist:
            print(f"\n📅 Weekly Distribution:")
            day_names = self.config['time_slot_config']['day_names']
            for day, count in daily_dist.items():
                if day < len(day_names):
                    print(f"   {day_names[day]}: {count} classes")

    def generate_pdf_timetables(self, output_dir: str = "professional_timetables"):
        """Generate professional PDF timetables with enhanced formatting"""
        if not REPORTLAB_AVAILABLE:
            print("📦 ReportLab not available. Install with: pip install reportlab")
            return

        print("📄 Generating professional PDF timetables...")
        os.makedirs(output_dir, exist_ok=True)

        time_config = self.config['time_slot_config']
        days = time_config['day_names']
        times = time_config['period_times']

        for division in self.divisions:
            filename = f"{output_dir}/timetable_{division.id}_{division.name.replace(' ', '_').replace('-', '_')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
            elements = []

            # Enhanced title with statistics
            styles = getSampleStyleSheet()
            title_text = f"PROFESSIONAL TIMETABLE - {division.name}<br/>Student Strength: {division.strength} | Semester: {division.semester}"
            title = Paragraph(title_text, styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 0.3*inch))

            # Create enhanced timetable table
            table_data = [['Time / Day'] + days]
            
            for period in range(self.periods_per_day):
                row = [times[period]]
                
                for day in range(self.days):
                    entry = self.division_schedule.get((day, period, division.id))
                    if entry:
                        subject = next(s for s in self.subjects if s.id == entry.subject_id)
                        teacher = next(t for t in self.teachers if t.id == entry.teacher_id)
                        room = next(r for r in self.rooms if r.id == entry.room_id)
                        
                        # Enhanced cell content with quality score
                        quality_indicator = "⭐" if hasattr(entry, 'assignment_score') and entry.assignment_score > 0.7 else ""
                        cell_text = f"{subject.name}\n{teacher.name}\n{room.id} {quality_indicator}"
                    else:
                        cell_text = ""
                    
                    row.append(cell_text)
                table_data.append(row)

            # Enhanced table styling
            table = Table(table_data, colWidths=[1.4*inch] + [1.6*inch]*self.days)
            table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                
                # Time column styling
                ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (0, -1), 9),
                
                # Content styling
                ('FONTSIZE', (1, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                
                # Alternating row colors
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.lightcyan]),
                
                # Lunch period highlighting
                ('BACKGROUND', (0, 4), (-1, 4), colors.lightyellow),  # Assuming lunch at period 3 (index 4)
            ]))

            elements.append(table)
            
            # Add statistics footer
            elements.append(Spacer(1, 0.3*inch))
            
            # Calculate division statistics
            division_entries = [e for e in self.schedule if e.division_id == division.id]
            if division_entries:
                avg_quality = sum(getattr(e, 'assignment_score', 0) for e in division_entries) / len(division_entries)
                unique_teachers = len(set(e.teacher_id for e in division_entries))
                unique_rooms = len(set(e.room_id for e in division_entries))
                
                stats_text = f"Statistics: {len(division_entries)} classes scheduled | {unique_teachers} teachers | {unique_rooms} rooms | Avg. Quality: {avg_quality:.2f}/100"
                stats = Paragraph(stats_text, styles['Normal'])
                elements.append(stats)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer_text = f"Generated on {timestamp} by Professional Timetable Generator"
            footer = Paragraph(footer_text, styles['Normal'])
            elements.append(Spacer(1, 0.1*inch))
            elements.append(footer)

            doc.build(elements)

        print(f"✅ Professional PDF timetables generated in '{output_dir}' directory")


def main():
    """Main function for professional timetable generation"""
    print("🎓 PROFESSIONAL UNIVERSITY TIMETABLE GENERATOR")
    print("=" * 60)
    print("Features: AI-based Teacher Assignment | Workload Optimization | Constraint Satisfaction")
    print("=" * 60)

    config_file = "pro_timetable_config.json"
    
    try:
        # Initialize generator
        generator = ProTimetableGenerator(config_file)
        
        # Generate data and timetable
        generator.generate_dummy_data()
        generator.generate_timetable()
        generator.print_detailed_statistics()
        
        # Generate PDFs ← THIS WAS MISSING!
        generator.generate_pdf_timetables()
        
        print("\n🎉 Professional timetable generation completed successfully!")
        print("💡 Features implemented:")
        print("   ✅ Expertise-based teacher assignment")
        print("   ✅ Intelligent workload balancing") 
        print("   ✅ Advanced constraint satisfaction")
        print("   ✅ Quality scoring and optimization")
        print("   ✅ Professional PDF generation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check your configuration file and try again.")


if __name__ == "__main__":
    main()
