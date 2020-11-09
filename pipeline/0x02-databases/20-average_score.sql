-- This script creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser(IN u_id INT)

BEGIN
UPDATE users SET average_score = (SELECT AVG(score) FROM corrections WHERE user_id = u_id) WHERE id = u_id;
END $$

DELIMITER ;
