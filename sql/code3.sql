CREATE TABLE Items
 ( item_id INTEGER NOT NULL,
   year    INTEGER NOT NULL,
   item_name CHAR(32) NOT NULL,
   price_tax_ex INTEGER NOT NULL,
   price_tax_in INTEGER NOT NULL,
   PRIMARY KEY (item_id, year));

INSERT INTO Items VALUES(100,   2001,   'カップ'        ,520,   546);


SELECT item_name, year, price_tax_ex AS price FROM Items WHERE year <= 2001
UNION ALL
SELECT item_name, year, price_tax_in AS price FROM Items WHERE year >= 2002;


SELECT item_name, year,
       CASE WHEN year <= 2001 THEN price_tax_ex
            WHEN year >= 2002 THEN price_tax_in END AS price
FROM Items;
  
CREATE TABLE Population
       (prefecture VARCHAR(32),
        sex CHAR(1),
        pop INTEGER,
	CONSTRAINT pk_pop PRIMARY KEY(prefecture, sex) );

INSERT INTO Population VALUES('徳島', '1', 60);

SELECT prefecture,
       SUM(CASE WHEN sex = '1' THEN pop ELSE 0 END) AS pop_men,
       SUM(CASE WHEN sex = '2' THEN pop ELSE 0 END) AS pop_wom
FROM Population
GROUP BY prefecture;

CREATE Table Employees
       (emp_id CHAR(3) NOT NULL,
        team_id INTEGER NOT NULL,
        emp_name CHAR(16) NOT NULL,
        team CHAR(16) NOT NULL,
       PRIMARY KEY(emp_id, team_id));

INSERT INTO Employees VALUES('201',  1,      'Joe',  '商品企画');

SELECT emp_name,
       CASE WHEN COUNT(*) = 1 THEN MAX(team)
       	    WHEN COUNT(*) = 2 THEN '2 operations'
	    WHEN COUNT(*) >= 3 THEN 'more than 3 ops'
       END AS team
FROM Employees
GROUP BY emp_name;
