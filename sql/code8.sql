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

CREATE TABLE Weights2
(class       INTEGER NOT NULL,
 student_id  CHAR(4) NOT NULL,
 weight      INTEGER NOT NULL,
 PRIMARY KEY(class, student_id));

INSERT INTO Weights2 VALUES(1, '100', 50);

SELECT class, student_id,
   (SELECT COUNT(*)
      FROM Weights2 W2
     WHERE (W2.class, W2.student_id)
             <= (W1.class, W1.student_id) ) AS seq
 FROM Weights2 W1;
 
