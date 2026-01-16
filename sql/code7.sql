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

