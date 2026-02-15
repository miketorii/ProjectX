CREATE TABLE Weights
(student_id CHAR(4) PRIMARY KEY,
 weight INTEGER);

INSERT INTO Weights VALUES('A100', 50);

SELECT student_id,
   ROW_NUMBER() OVER (ORDER BY student_id) AS seq
 FROM Weights;

SELECT student_id,
   (SELECT COUNT(*) FROM Weights W2
      WHERE W2.student_id <= W1.student_id) AS seq
 FROM Weights W1;

