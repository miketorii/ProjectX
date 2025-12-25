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

