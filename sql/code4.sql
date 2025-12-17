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


