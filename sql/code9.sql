CREATE TABLE OmitTbl
(keycol CHAR(8) NOT NULL,
 seq     INTEGER NOT NULL,
 val     INTEGER,
   CONSTRAINT pk_OmitTbl PRIMARY KEY (keycol, seq));

INSERT INTO OmitTbl VALUES ('A', 1, 50);

UPDATE OmitTbl
 SET val = (SELECT val FROM OmitTbl O1
              WHERE O1.keycol = OmitTbl.keycol
                AND O1.seq = (SELECT MAX(seq) FROM OmitTbl O2
                                WHERE O2.keycol = OmitTbl.keycol
                                  AND O2.seq < OmitTbl.seq
                                  AND O2.val IS NOT NULL))
 WHERE val IS NULL;

UPDATE OmitTbl
 SET val = CASE WHEN val
                 = (SELECT val FROM OmitTbl O1 
                       WHERE O1.keycol = OmitTbl.keycol
                         AND O1.seq = (SELECT MAX(seq) FROM OmitTbl O2
                                         WHERE O2.keycol = OmitTbl.keycol
                                           AND O2.seq < OmitTbl.seq))
           THEN NULL
           ELSE val END;

CREATE TABLE ScoreRows
(student_id CHAR(4)   NOT NULL,
 subject    VARCHAR(8) NOT NULL,
 score      INTEGER,
  CONSTRAINT pk_ScoreRows PRIMARY KEY(student_id, subject));

CREATE TABLE ScoreCols
(student_id CHAR(4)   NOT NULL,
 score_en   INTEGER,
 score_nl   INTEGER,
 score_mt   INTEGER,
  CONSTRAINT pk_ScoreCols PRIMARY KEY(student_id));

INSERT INTO ScoreRows VALUES('A001', '英語', 100);

INSERT INTO ScoreCols VALUES('A001', NULL, NULL, NULL);

UPDATE ScoreCols
 SET score_en = (SELECT score FROM ScoreRows SR
                   WHERE SR.student_id = ScoreCols.student_id
                     AND subject = '英語'),
     score_nl = (SELECT score FROM ScoreRows SR
                   WHERE SR.student_id = ScoreCols.student_id
                     AND subject = '国語'),
     score_mt = (SELECT score FROM ScoreRows SR
                   WHERE SR.student_id = ScoreCols.student_id
                     AND subject = '数学');


UPDATE ScoreCols
  SET (score_en, score_nl, score_mt)
    = (SELECT MAX(CASE WHEN subject='英語' THEN score ELSE NULL END) AS score_en,
              MAX(CASE WHEN subject='国語' THEN score ELSE NULL END) AS score_nl,
              MAX(CASE WHEN subject='数学' THEN score ELSE NULL END) AS score_mt
  FROM ScoreRows SR
  WHERE SR.student_id = ScoreCols.student_id);

CREATE TABLE ScoreColsNN
(student_id CHAR(4) NOT NULL,
 score_en   INTEGER NOT NULL,
 score_nl   INTEGER NOT NULL,
 score_mt   INTEGER NOT NULL,
   CONSTRAINT pk_ScoreColsNN PRIMARY KEY (student_id));

INSERT INTO ScoreColsNN VALUES ('A001', 0, 0, 0);

UPDATE ScoreColsNN
 SET score_en = COALESCE((SELECT score FROM ScoreRows
                            WHERE student_id = ScoreColsNN.student_id
                              AND subject = '英語'), 0),
     score_nl = COALESCE((SELECT score FROM ScoreRows
                            WHERE student_id = ScoreColsNN.student_id
                              AND subject = '国語'), 0),
     score_mt = COALESCE((SELECT score FROM ScoreRows
                            WHERE student_id = ScoreColsNN.student_id
                              AND subject = '数学'), 0)
   WHERE EXISTS (SELECT * FROM ScoreRows
                   WHERE student_id = ScoreColsNN.student_id);

MERGE INTO ScoreColsNN AS T
  USING (SELECT student_id,
                COALESCE(MAX(CASE WHEN subject='英語'
                                  THEN score
                                  ELSE NULL END), 0) AS score_en,
                COALESCE(MAX(CASE WHEN subject='国語'
                                  THEN score
                                  ELSE NULL END), 0) AS score_nl,
                COALESCE(MAX(CASE WHEN subject='数学'
                                  THEN score
                                  ELSE NULL END), 0) AS score_mt
            FROM ScoreRows
           GROUP BY student_id) AS SR
       ON (T.student_id = SR.student_id)
     WHEN MATCHED THEN
          UPDATE SET score_en = SR.score_en,
                     score_nl = SR.score_nl,
                     score_mt = SR.score_mt;

UPDATE ScoreRows
  SET score = (SELECT CASE ScoreRows.subject
                      WHEN '英語' THEN score_en
                      WHEN '国語' THEN score_nl
                      WHEN '数学' THEN score_mt
                      ELSE NULL END
                 FROM ScoreCols
                WHERE student_id = ScoreRows.student_id);

CREATE TABLE Stocks
(brand     VARCHAR(8) NOT NULL,
 sale_date DATE   NOT NULL,
 price     INTEGER    NOT NULL,
    CONSTRAINT pk_Stocks PRIMARY KEY (brand, sale_date));

INSERT INTO Stocks VALUS('A鉄鋼', '2008-07-01', 1000);

CREATE TABLE Stocks2
(brand     VARCHAR(8) NOT NULL,
 sale_date DATE   NOT NULL,
 price     INTEGER    NOT NULL,
 trend     CHAR(3),
    CONSTRAINT pk_Stocks2 PRIMARY KEY (brand, sale_date));

INSERT INTO Stocks2
SELECT brand, sale_date, price,
       CASE SIGN(price-MAX(price) OVER (PARTITION BY brand
                                            ORDER BY sale_date
                                        ROWS BETWEEN 1 PRECEDING
                                                 AND 1 PRECEDING))
            WHEN -1 THEN '↓'
            WHEN 0  THEN '→'
            WHEN 1  THEN '↑'
            ELSE NULL
       END
 FROM Stocks S2;
 
CREATE TABLE Orders
( order_id INTEGER NOT NULL,
  order_shop VARCHAR(32) NOT NULL,
  order_name VARCHAR(32) NOT NULL,
  order_date DATE,
  PRIMARY KEY (order_id) );

INSERT INTO Orders VALUES (10000, '東京', 'Goto', '2011/8/22');

CREATE TABLE OrderReceipts
( order_id INTEGER NOT NULL,
  order_receipt_id INTEGER NOT NULL,
  item_group VARCHAR(32) NOT NULL,
  delivery_date DATE NOT NULL,
  PRIMARY KEY (order_id, order_receipt_id));

INSERT INTO OrderReceipts VALUES (10000, 1, '食器', '2011/8/24');

SELECT O.order_id, O.order_name,
       ORC.delivery_date - O.order_date AS diff_days
  FROM Orders O
          INNER JOIN OrderReceipts ORC
             ON O.order_id = ORC.order_id
  WHERE ORC.delivery_date - O.order_date >= 3;


SELECT O.order_id,
       MAX(O.order_name),
       MAX(ORC.delivery_date - O.order_date) AS max_diff_days
  FROM Orders O
         INNER JOIN OrderReceipts ORC
            ON O.order_id = ORC.order_id
 WHERE ORC.delivery_date - O.order_date >= 3
 GROUP BY O.order_id;

