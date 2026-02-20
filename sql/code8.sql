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
 
SELECT class, student_id,
    ROW_NUMBER() OVER (PARTITION BY class ORDER BY student_id) AS seq
  FROM Weights2;
  
SELECT class, student_id,
     (SELECT COUNT(*) FROM Weights2 W2
        WHERE (W2.class, W2.student_id) <= (W1.class, W1.student_id)
     ) AS seq
 FROM Weights2 W1;
 
CREATE TABLE Weights3
(class      INTEGER NOT NULL,
 student_id CHAR(4) NOT NULL,
 weight     INTEGER NOT NULL,
 seq        INTEGER NULL,
    PRIMARY KEY(class, student_id));

INSERT INTO Weights3 VALUES(1, '100', 50, NULL);

UPDATE Weights3
 SET seq = (SELECT seq
          FROM ( SELECT class, student_id,
                     ROW_NUMBER() 
                      OVER (PARTITION BY class
                              ORDER BY student_id) AS seq
                 FROM Weights3) SeqTbl
         WHERE Weights3.class = SeqTbl.class
           AND Weights3.student_id = SeqTbl.student_id );

UPDATE Weights3
  SET seq = (SELECT COUNT(*) FROM Weights3 W2
               WHERE W2.class = Weights3.class
                 AND W2.student_id <= Weights3.student_id);
