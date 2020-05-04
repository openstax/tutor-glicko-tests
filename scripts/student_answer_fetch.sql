SELECT er.research_identifier, 
       cpc.id as "course_id",
       COALESCE(cpc.is_college, True) as "is_college", 
       cms.course_membership_period_id AS "section_id",
       tt.id AS "assignment_id",
       tte.answer_id = tte.correct_answer_id AS "response_correct",
       ce.group_uuid||'#'||tte.question_index as "question_id",
       tt.last_worked_at AS "response_time"
FROM tasks_tasked_exercises tte 
JOIN tasks_task_steps tts
        ON (tts.tasked_id = tte.id AND tts.tasked_type = 'Tasks::Models::TaskedExercise')
JOIN tasks_tasks tt
        ON tts.tasks_task_id = tt.id
JOIN tasks_taskings tti
        ON tti.tasks_task_id = tt.id
JOIN entity_roles er
        ON tti.entity_role_id = er.id
JOIN user_profiles up
        ON er.user_profile_id = up.id
JOIN course_membership_students cms
	ON cms.entity_role_id = er.id
JOIN course_profile_courses cpc
	ON cms.course_profile_course_id = cpc.id
JOIN content_exercises ce
	ON tte.content_exercise_id = ce.id
WHERE answer_id IS NOT NULL 
    AND tt.completed_steps_count = tt.steps_count
    AND tt.last_worked_at > '2019/01/01'::timestamptz
    AND NOT cpc.is_preview
    AND NOT cpc.is_test
ORDER BY response_time, research_identifier, assignment_id

;
