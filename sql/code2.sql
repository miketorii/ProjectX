CREATE TABLE Address
(      name VARCHAR(32) NOT NULL,
      phone_nbr	VARCHAR(32),
      address	VARCHAR(32) NOT NULL,
      sex	CHAR(4) NOT NULL,
      age	INTEGER NOT NULL,
      PRIMARY KEY (name));

INSERT INTO Address VALUES('小川','080-333-1234','東京都','男',30);

SELECT name, phone_nbr, address, sex, age FROM Address;

SELECT name, address FROM Address WHERE address = '千葉県';

SELECT name, age FROM Address WHERE age >= 30;

SELECT name, address FROM Address WHERE address <> '東京都';

SELECT name, address FROM Address WHERE  address='東京都' OR address='福島県' OR address='千葉県';

SELECT name, address FROM Address WHERE address IN ('東京都','福島県','千葉県');

SELECT name, phone_nbr FROM Address WHERE phone_nbr IS NULL;

SELECT sex, COUNT(*) FROM Address GROUP BY sex;

SELECT address, COUNT(*) FROM Address GROUP BY address;

SELECT COUNT(*) FROM Address;

SELECT address, COUNT(*) FROM Address GROUP BY address HAVING COUNT(*)=1;

SELECT name, phone_nbr, address, sex, age FROM Address ORDER BY age DESC;

CREATE VIEW CountAddress (v_address, cnt) AS SELECT address, COUNT(*) FROM Address GROUP BY address;
SELECT v_address, cnt FROM CountAddress;

CREATE TABLE Address2(name VARCHAR(32) NOT NULL, phone_nbr VARCHAR(32), address VARCHAR(32) NOT NULL, sex CHAR(4) NOT NULL, age INTEGER NOT NULL, PRIMARY KEY (name) );

INSERT INTO Address2 VALUES('小川', '080-333-XXXX', '東京都', '男', 30);

SELECT * FROM Address2

select * from Address
UNION
select * from Address2;

SELECT * FROM Address
INTERSECT
SELECT * FROM Address2;

SELECT * FROM Address
EXCEPT
SELECT * FROM Address2;

UPDATE Address2 
SET phone_nbr='080-3333-XXXX'
WHERE name='小川'

SELECT address,
       COUNT(*) OVER(PARTITION BY address)
FROM Address;

SELECT name,
       age,
       RANK() OVER(ORDER BY age DESC) AS rnk
FROM Address;
