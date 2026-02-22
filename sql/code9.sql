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

