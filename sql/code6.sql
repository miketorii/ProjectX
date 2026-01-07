CREATE TABLE Employees2
(emp_id CHAR(8),
 emp_name VARCHAR(32),
 dept_id CHAR(2),
    CONSTRAINT pk_emp PRIMARY KEY(emp_id));

CREATE TABLE Departments
(dept_id CHAR(2),
 dept_name VARCHAR(32),
    CONSTRAINT pk_dep PRIMARY KEY(dept_id));

CREATE INDEX idx_dept_id ON Employees2(dept_id);

INSERT INTO Employees2 VALUES('001',	'石田',	  '10');

INSERT INTO Departments VALUES('10',	'総務');

SELECT * FROM Employees2
  CROSS JOIN Departments;

SELECT E.emp_id, E.emp_name, e.dept_id, D.dept_name
  FROM Employees2 E INNER JOIN Departments D
    ON E.dept_id = D.dept_id;

SELECT E.emp_id, E.emp_name, E.dept_id,
 (SELECT D.dept_name FROM Departments D WHERE E.dept_id = D.dept_id) AS dept_name
 FROM Employees2 E;
 
SELECT E.emp_id, E.emp_name, E.dept_id, D.dept_name
  FROM Departments D LEFT OUTER JOIN Employees2 E
   ON D.dept_id = E.dept_id;
