CREATE TABLE Sales
(company CHAR(1) NOT NULL,
 year	 INTEGER NOT NULL,
 sale	 INTEGER NOT NULL,
  CONSTRAINT pk_sales PRIMARY KEY (company, year));

INSERT INTO Sales VALUES('A', 2002, 50);

CREATE TABLE Sales2
(company CHAR(1) NOT NULL,
 year	 INTEGER NOT NULL,
 sale	 INTEGER NOT NULL,
 var     CHAR(1),
  CONSTRAINT pk_sales2 PRIMARY KEY (company, year));

CREATE TABLE Foo
( p_key INTEGER PRIMARY KEY,
  col_a INTEGER
);

INSERT INTO Foo VALUES( 1, 100 );
INSERT INTO Foo VALUES( 2, 200 );
INSERT INTO Foo VALUES( 3, 300 );
INSERT INTO Foo VALUES( 4, 400 );

SELECT col_a FROM Foo WHERE p_key = 1

CREATE OR REPLACE PROCEDURE PROC_INSERT_VAR()
LANGUAGE plpgsql
AS $$
DECLARE
    c_sales CURSOR FOR
        SELECT company, year, sale
          FROM Sales
         ORDER BY company, year;

    rec_sales    RECORD;
    i_pre_sale   INTEGER := 0;
    v_pre_company TEXT := '*'; 
    c_var        VARCHAR(1) := '*';

BEGIN
    FOR rec_sales IN 
        SELECT company, year, sale FROM Sales ORDER BY company, year 
    LOOP

        IF (v_pre_company = rec_sales.company) THEN
            IF (i_pre_sale < rec_sales.sale) THEN
                c_var := '+';
            ELSIF (i_pre_sale > rec_sales.sale) THEN
                c_var := '-';
            ELSE
                c_var := '=';
            END IF;
        ELSE
            c_var := NULL;
        END IF;

        INSERT INTO Sales2 (company, year, sale, var) 
        VALUES (rec_sales.company, rec_sales.year, rec_sales.sale, c_var);

        v_pre_company := rec_sales.company;
        i_pre_sale    := rec_sales.sale;

    END LOOP;

END;
$$;


CALL PROC_INSERT_VAR();

INSERT INTO Sales2
SELECT company, year, sale,
	CASE SIGN(sale - MAX(sale) 
			OVER( PARTITION BY company ORDER BY year
				ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) )
	WHEN 0 THEN '='
	WHEN 1 THEN '+'
	WHEN -1 THEN '-'
	ELSE NULL END AS var
  FROM Sales;

CREATE TABLE PostalCode
(pcode CHAR(7),
 district_name VARCHAR(256),
	CONSTRAINT pk_pcode PRIMARY KEY(pcode));

INSERT INTO PostalCode VALUES('4130001', '静岡県熱海市泉');

SELECT pcode, district_name,
       CASE WHEN pcode = '4130033' THEN 0
            WHEN pcode LIKE '413003%' THEN 1
            WHEN pcode LIKE '41300%'  THEN 2
            WHEN pcode LIKE '4130%'   THEN 3
            WHEN pcode LIKE '413%'    THEN 4
            WHEN pcode LIKE '41%'     THEN 5
            WHEN pcode LIKE '4%'      THEN 6
       ELSE NULL END AS rank
 FROM PostalCode;

SELECT pcode,
       district_name
  FROM PostalCode
 WHERE CASE WHEN pcode = '4130033' THEN 0
             WHEN pcode LIKE '413003%' THEN 1
             WHEN pcode LIKE '41300%'  THEN 2
             WHEN pcode LIKE '4130%'   THEN 3
             WHEN pcode LIKE '413%'    THEN 4
             WHEN pcode LIKE '41%'     THEN 5
             WHEN pcode LIKE '4%'      THEN 6
             ELSE NULL END = 
		( SELECT MIN(CASE WHEN pcode = '4130033' THEN 0
                                  WHEN pcode LIKE '413003%' THEN 1
                                  WHEN pcode LIKE '41300%'  THEN 2
                                  WHEN pcode LIKE '4130%'   THEN 3
                                  WHEN pcode LIKE '413%'    THEN 4
                                  WHEN pcode LIKE '41%'     THEN 5
                                  WHEN pcode LIKE '4%'      THEN 6
                                  ELSE NULL END) 
		  FROM PostalCode);

SELECT * FROM PostalCode;

SELECT pcode,
       district_name
  FROM (SELECT pcode,
               district_name,
               CASE WHEN pcode = '4130033' THEN 0
                    WHEN pcode LIKE '413003%' THEN 1
                    WHEN pcode LIKE '41300%'  THEN 2
                    WHEN pcode LIKE '4130%'   THEN 3
                    WHEN pcode LIKE '413%'    THEN 4
                    WHEN pcode LIKE '41%'     THEN 5
                    WHEN pcode LIKE '4%'      THEN 6
                    ELSE NULL END AS hit_code,
               MIN(CASE WHEN pcode = '4130033' THEN 0
                        WHEN pcode LIKE '413003%' THEN 1
                        WHEN pcode LIKE '41300%'  THEN 2
                        WHEN pcode LIKE '4130%'   THEN 3
                        WHEN pcode LIKE '413%'    THEN 4
                        WHEN pcode LIKE '41%'     THEN 5
                        WHEN pcode LIKE '4%'      THEN 6
                        ELSE NULL END)
               OVER(ORDER BY CASE WHEN pcode = '4130033' THEN 0
                                  WHEN pcode LIKE '413003%' THEN 1
                                  WHEN pcode LIKE '41300%'  THEN 2
                                  WHEN pcode LIKE '4130%'   THEN 3
                                  WHEN pcode LIKE '413%'    THEN 4
                                  WHEN pcode LIKE '41%'     THEN 5
                                  WHEN pcode LIKE '4%'      THEN 6
                                  ELSE NULL END) AS min_code 
            FROM PostalCode) Foo
WHERE hit_code=min_code;


CREATE TABLE PostalHistory
(name CHAR(1),
 pcode CHAR(7),
 new_pcode CHAR(7),
   CONSTRAINT pk_name_pcode PRIMARY KEY(name, pcode) );

INSERT INTO PostalHistory VALUES ('A', '4130001', '4130002');


WITH RECURSIVE Explosion (name, pcode, new_pcode, depth)
AS
(SELECT name, pcode, new_pcode, 1 
   FROM PostalHistory
     WHERE name = 'A' AND new_pcode IS NULL
 UNION
 SELECT Child.name, Child.pcode, Child.new_pcode, depth+1 
   FROM Explosion AS Parent, PostalHistory AS Child
     WHERE Parent.pcode = Child.new_pcode AND Parent.name = Child.name)
SELECT name, pcode, new_pcode
  FROM Explosion
    WHERE depth = (SELECT MAX(depth) FROM Explosion);




CREATE TABLE PostalHistory2
(name CHAR(1),
 pcode CHAR(7),
 lft REAL NOT NULL,
 rgt REAL NOT NULL,
	CONSTRAINT pk_name_pcode2 PRIMARY KEY(name, pcode),
	CONSTRAINT ug_name_lft UNIQUE(name, lft),
	CONSTRAINT ug_name_rgt UNIQUE(name, rgt),
	CHECK(lft<rgt));

INSERT INTO PostalHistory2 VALUES('A', '4130001', 0, 27);


SELECT name, pcode
  FROM PostalHistory2 PH1
 WHERE name = 'A' AND NOT EXISTS (
	SELECT * FROM PostalHistory2 PH2
	 WHERE PH2.name = 'A' AND PH1.lft > PH2.lft
);
