CREATE TABLE Shops (
 shop_id    CHAR(5) NOT NULL,
 shop_name  VARCHAR(64),
 rating     INTEGER,
 area       VARCHAR(64),
   CONSTRAINT pk_shops PRIMARY KEY (shop_id));

INSERT INTO Shops (shop_id, shop_name, rating, area) VALUES ('00001', '○○商店', 3, '北海道');

SELECT * FROM Shops;

SELECT * FROM Shops WHERE shop_id = '00050';
