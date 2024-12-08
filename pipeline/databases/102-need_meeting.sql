-- Creates a view need_meeting that lists students needing a meeting
CREATE VIEW need_meeting AS
SELECT name FROM students
WHERE score < 80
AND (last_meeting IS NULL 
     OR last_meeting < DATE_SUB(CURDATE(), INTERVAL 1 MONTH));
