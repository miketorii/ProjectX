CREATE TABLE NonAggTbl
(id VARCHAR(32) NOT NULL,
 data_type CHAR(1) NOT NULL,
 data_1	INTEGER,
 data_2	INTEGER,
 data_3	INTEGER,
 data_4	INTEGER,
 data_5	INTEGER,
 data_6	INTEGER );

DELETE FROM NonAgTbl;
INSERT INTO NonAgTbl VALUES('Jim','A',100,10,34,346,54,NULL);


SELECT id, data_1, data_2
 FROM NonAggTbl
WHERE id = 'Jim' AND data_type = 'A';

SELECT id, data_3, data_4, data_5
 FROM NonAggTbl
WHERE id = 'Jim' AND data_type = 'B';

SELECT id, data_6
 FROM NonAggTbl
WHERE id = 'Jim' AND data_type = 'C';

SELECT id,
 	MAX(CASE WHEN data_type = 'A' THEN data_1 ELSE NULL END) AS data_1,
 	MAX(CASE WHEN data_type = 'A' THEN data_2 ELSE NULL END) AS data_2,
 	MAX(CASE WHEN data_type = 'B' THEN data_3 ELSE NULL END) AS data_3,
 	MAX(CASE WHEN data_type = 'B' THEN data_4 ELSE NULL END) AS data_4,
 	MAX(CASE WHEN data_type = 'B' THEN data_5 ELSE NULL END) AS data_5,
 	MAX(CASE WHEN data_type = 'C' THEN data_6 ELSE NULL END) AS data_6
 FROM NonAggTbl
GROUP BY id;

CREATE TABLE PriceByAge
(product_id 	VARCHAR(32) NOT NULL,
 low_age	INTEGER NOT NULL,
 high_age	INTEGER NOT NULL,
 price		INTEGER NOT NULL,
 PRIMARY KEY (product_id, low_age),
   CHECK (low_age < high_age));

INSERT INTO PriceByAge VALUES('製品1', 0, 50, 2000);

SELECT product_id
 FROM PriceByAge
 GROUP BY product_id
HAVING SUM(high_age - low_age + 1) = 101;

CREATE TABLE HotelRooms
(room_nbr	INTEGER,
 start_date	DATE,
 end_date	DATE,
	PRIMARY KEY(room_nbr, start_date));

INSERT INTO HotelRooms VALUES(101, '2008-02-01', '2008-02-06');

SELECT room_nbr,
	SUM(end_date - start_date) AS working_days
  FROM HotelRooms
 GROUP BY room_nbr
HAVING SUM(end_date - start_date) >= 10;

CREATE TABLE Persons
(name	VARCHAR(8) NOT NULL,
 age	INTEGER NOT NULL,
 height	FLOAT NOT NULL,
 weight	FLOAT NOT NULL,		
 PRIMARY KEY (name));

INSERT INTO Persons VALUES('Anderson', 30, 188, 90);

SELECT SUBSTRING(name, 1, 1) AS label,
	COUNT(*)
  FROM Persons
 GROUP BY SUBSTRING(name, 1, 1);

SELECT 	CASE 	WHEN age < 20 THEN 'Child'
		WHEN age BETWEEN 20 AND 69 THEN 'Adult'
		WHEN age >= 70 THEN 'Senior'
	ELSE NULL END AS age_class,
	COUNT(*)
  FROM Persons
 GROUP BY CASE 	WHEN age < 20 THEN 'Child'
		WHEN age BETWEEN 20 AND 69 THEN 'Adult'
		WHEN age >= 70 THEN 'Senior'
	ELSE NULL END;

SELECT CASE 	WHEN weight / POWER(height/100, 2) < 18.5 THEN 'Thin'
		WHEN 18.5 <= weight / POWER(height/100, 2)
			AND weight / POWER(height/100, 2) < 25 THEN 'Standard'
		WHEN 25 <= weight / POWER(height/100, 2) THEN 'Fat'
		ELSE NULL END AS bmi,
		COUNT(*)
  FROM Persons
 GROUP BY CASE 	WHEN weight / POWER(height/100, 2) < 18.5 THEN 'Thin'
		WHEN 18.5 <= weight / POWER(height/100, 2)
			AND weight / POWER(height/100, 2) < 25 THEN 'Standard'
		WHEN 25 <= weight / POWER(height/100, 2) THEN 'Fat'
		ELSE NULL END;


SELECT name, age,
	CASE 	WHEN age < 20 THEN 'child'
		WHEN age BETWEEN 20 AND 69 THEN 'adult'
		WHEN age >= 70 THEN 'senior'
	ELSE NULL END AS age_class,
	RANK() OVER( PARTITION BY CASE 	WHEN age < 20 THEN 'child'
					WHEN age BETWEEN 20 AND 69 THEN 'adult'
					WHEN age >= 70 THEN 'senior'
				   ELSE NULL END
			ORDER BY age ) AS age_rank_in_class
  FROM Persons
 ORDER BY age_class, age_rank_in_class;
