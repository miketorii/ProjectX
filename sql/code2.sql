CREATE TABLE Address
(      name VARCHAR(32) NOT NULL,
      phone_nbr	VARCHAR(32),
      address	VARCHAR(32) NOT NULL,
      sex	CHAR(4) NOT NULL,
      age	INTEGER NOT NULL,
      PRIMARY KEY (name));

INSERT INTO Address VALUES('小川','080-333-1234','東京都','男',30);

SELECT name, phone_nbr, address, sex, age FROM Address;

SELECT name, age FROM Address WHERE age >= 30;

