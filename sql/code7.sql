CREATE TABLE Receipts
(cust_id CHAR(1) NOT NULL,
 seq     INTEGER NOT NULL,
 price   INTEGER NOT NULL,
   PRIMARY KEY (cust_id, seq) );

INSERT INTO Receipts VALUES ('A', 1, 500 );

SELECT R1.cust_id, R1.seq, R1.price
  FROM Receipts R1
    INNER JOIN ( SELECT cust_id, MIN(seq) AS min_seq
                   FROM Receipts
                    GROUP BY cust_id) R2
  ON R1.cust_id = R2.cust_id
 AND R1.seq = R2.min_seq;
 
SELECT cust_id, seq, price
  FROM Receipts R1
 WHERE seq = ( SELECT MIN(seq)
                 FROM Receipts R2
                WHERE R1.cust_id = R2.cust_id);
		
SELECT cust_id, seq, price
  FROM (SELECT cust_id, seq, price, 
               ROW_NUMBER()
                OVER (PARTITION BY cust_id ORDER BY seq) AS row_seq 
           FROM Receipts ) WORK
 WHERE WORK.row_seq = 1;

SELECT TMP_MIN.cust_id,
       TMP_MIN.price - TMP_MAX.price AS diff
  FROM (

          SELECT R1.cust_id, R1.seq, R1.price
            FROM Receipts R1
           INNER JOIN (

                        SELECT cust_id, MIN(seq) AS min_seq
                          FROM Receipts
                         GROUP BY cust_id 

                       ) R2
           ON R1.cust_id = R2.cust_id AND R1.seq = R2.min_seq

       ) TMP_MIN

       INNER JOIN (

                     SELECT R3.cust_id, R3.seq, R3.price
                       FROM Receipts R3
                     INNER JOIN(

                                   SELECT cust_id, MAX(seq) AS min_seq
                                    FROM Receipts
                                   GROUP BY cust_id 

                                ) R4
                     ON R3.cust_id = R4.cust_id AND R3.seq = R4.min_seq

       ) TMP_MAX

  ON TMP_MIN.cust_id = TMP_MAX.cust_id;

SELECT cust_id, 
       SUM(CASE WHEN min_seq=1 THEN price ELSE 0 END) 
       - SUM(CASE WHEN max_seq=1 THEN price ELSE 0 END) AS diff
  FROM (SELECT cust_id, price,
               ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY seq) AS min_seq,
               ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY seq DESC) AS max_seq
        FROM Receipts) WORK
 WHERE WORK.min_seq=1 OR WORK.max_seq=1
 GROUP BY cust_id;

CREATE TABLE Companies
(co_cd	CHAR(3) NOT NULL,
 district CHAR(1) NOT NULL,
	CONSTRAINT pk_Companies PRIMARY KEY (co_cd));

INSERT INTO Companies VALUES('001', 'A');

CREATE TABLE Shops2
(co_cd CHAR(3) NOT NULL,
 shop_id CHAR(3) NOT NULL,
 emp_nbr INTEGER NOT NULL,
 main_flg CHAR(1) NOT NULL,
	PRIMARY KEY (co_cd, shop_id));

INSERT INTO Shops2 VALUES('001', '1', 300, 'Y');


SELECT C.co_cd, C.district,
	SUM(emp_nbr) AS sum_emp
  FROM Companies C
	INNER JOIN
	  Shops2 S
   ON C.co_cd = S.co_cd
 WHERE main_flg = 'Y'
 GROUP BY C.co_cd;

SELECT C.co_cd, C.district, sum_emp
  FROM Companies C
	INNER JOIN
	 (SELECT co_cd, SUM(emp_nbr) AS sum_emp
              FROM Shops2
            WHERE main_flg = 'Y'
            GROUP BY co_cd) CSUM
   ON C.co_cd = CSUM.co_cd;
