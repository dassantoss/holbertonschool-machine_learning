-- Creates a function to divide two numbers
DELIMITER $$

DROP FUNCTION IF EXISTS SafeDiv $$

CREATE FUNCTION SafeDiv(a INT, b INT) 
RETURNS FLOAT
DETERMINISTIC
BEGIN
    -- Check if the divisor is zero
    IF b = 0 THEN
        RETURN 0; -- Return 0 if divisor is 0
    ELSE
        RETURN a / b; -- Otherwise, return the division result
    END IF;
END $$

DELIMITER ;
