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
