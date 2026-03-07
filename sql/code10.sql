CREATE TABLE Orders2
(order_id CHAR(8) NOT NULL,
 shop_id  CHAR(4) NOT NULL,
 shop_name VARCHAR(256) NOT NULL,
 receive_date DATE NOT NULL,
 process_flg CHAR(1) NOT NULL,
   CONSTRAINT pk_Orders PRIMARY KEY(order_id)
);

SELECT order_id, receive_date FROM Orders2
 WHERE process_flg = '1';

SELECT order_id, shop_name FROM Orders2
 WHERE receive_date BETWEEN '2026-03-01' AND '2026-03-05';

SELECT COUNT(*) FROM Orders2 WHERE shop_id='S001';

