-- Creates stored procedure ComputeAverageWeightedScoreForUser
DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id INT)
BEGIN
    DECLARE weighted_avg FLOAT;
    
    -- Calculate weighted average
    SELECT SUM(corrections.score * projects.weight) / SUM(projects.weight)
    INTO weighted_avg
    FROM corrections
    INNER JOIN projects ON corrections.project_id = projects.id
    WHERE corrections.user_id = user_id;
    
    -- Update user's average_score
    UPDATE users
    SET average_score = weighted_avg
    WHERE id = user_id;
END $$

DELIMITER ;
