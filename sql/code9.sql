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

