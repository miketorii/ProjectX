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

SELECT AVG(weight)
 FROM (SELECT W1.weight
         FROM Weights W1, Weights W2
        GROUP BY W1.weight
       HAVING SUM(CASE WHEN W2.weight >= W1.weight THEN 1 ELSE 0 END) >= COUNT(*)/2
          AND SUM(CASE WHEN W2.weight <= W1.weight THEN 1 ELSE 0 END) >= COUNT(*)/2 ) TMP;

SELECT AVG(weight) AS median
  FROM ( SELECT weight,
                ROW_NUMBER() OVER (ORDER BY weight ASC, student_id ASC) AS hi,
                ROW_NUMBER() OVER (ORDER BY weight DESC, student_id DESC) AS lo
          FROM Weights) TMP
 WHERE hi IN (lo, lo+1, lo-1);
 
SELECT AVG(weight)
  FROM( SELECT weight, 
               2*ROW_NUMBER() OVER(ORDER BY weight) - COUNT(*) OVER() AS diff
        FROM Weights) TMP
 WHERE diff BETWEEN 0 AND 2;

CREATE TABLE Numbers
(num  INTEGER PRIMARY KEY);

INSERT INTO Numbers VALUES(1);

SELECT (N1.num + 1) AS gap_start,
       '〜',
       (MIN(N2.num)-1) AS gap_end
   FROM Numbers N1 INNER JOIN Numbers N2
     ON N2.num > N1.num
  GROUP BY N1.num
 HAVING (N1.num+1) < MIN(N2.num);
