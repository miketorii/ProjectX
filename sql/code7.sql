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
 
