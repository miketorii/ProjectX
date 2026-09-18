select メモ from 家計簿
select 日付, 費目, 出金額 from 家計簿;
select * from 家計簿;
select 日付, 費目, 出金額 from 家計簿 where 出金額 > 3000;
insert into 家計簿 values ('2018-02-25','居住費','3月の家賃',0,85000);
update 家計簿 set 出金額=90000 where 日付='2024-02-11';
delete from 家計簿 where 日付='2024-02-14';




