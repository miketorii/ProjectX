CREATE TABLE Shops (
 shop_id    CHAR(5) NOT NULL,
 shop_name  VARCHAR(64),
 rating     INTEGER,
 area       VARCHAR(64),
   CONSTRAINT pk_shops PRIMARY KEY (shop_id));

INSERT INTO Shops (shop_id, shop_name, rating, area) VALUES ('00001', '○○商店', 3, '北海道');

#####################################################3

SELECT * FROM Shops;

SELECT * FROM Shops WHERE shop_id = '00050';

#####################################################3

CREATE TABLE Reservations (
 reserve_id    INTEGER  NOT NULL,
 shop_id       CHAR(5),
 reserve_name  VARCHAR(64),
   CONSTRAINT pk_reservations PRIMARY KEY (reserve_id));

INSERT INTO Reservations (reserve_id, shop_id, reserve_name) VALUES (1, '00001', 'Aさん');
