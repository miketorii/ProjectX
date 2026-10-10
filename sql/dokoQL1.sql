/*************************************************/
/*                  chaper 1                     */
/*************************************************/

select メモ from 家計簿
select 日付, 費目, 出金額 from 家計簿;
select * from 家計簿;
select 日付, 費目, 出金額 from 家計簿 where 出金額 > 3000;
insert into 家計簿 values ('2018-02-25','居住費','3月の家賃',0,85000);
update 家計簿 set 出金額=90000 where 日付='2024-02-11';
delete from 家計簿 where 日付='2024-02-14';

/*************************************************/
/*                  chaper 2                     */
/*************************************************/

select 費目 as ITEM, 入金額 as RECEIVE, 出金額 as PAY
 from 家計簿 as MONEYBOOK
where 費目= '給料';

select * from 家計簿 where 出金額>0;
select * from 家計簿 where 入金額 is NOT NULL;
select * from 家計簿 where メモ like '%1月%';
select * from 家計簿 where 出金額 between 100 and 3000;
select * from 家計簿 where 費目 in ('食費','交際費');
select * from 家計簿 where 費目 not in ('食費','交際費');

select * from 家計簿 where 出金額 < any(array[2800,4000,5000]);
select * from 家計簿 where 出金額 < any(values (2800),(4000),(5000));
select * from 家計簿 where 出金額 < all(values (2800),(4000),(5000));
select * from 家計簿 where 入金額 <> 出金額;

/*************************************************/
/*                  chaper 3                     */
/*************************************************/

insert into 家計簿 values ('2024-03-11','交際費','テスト用',2000,3000);
update 家計簿 set 出金額=5000 where 費目='交際費' and 入金額=2000;
select * from 家計簿 where 入金額>0 or 出金額>=5000;
select * from 家計簿 where 日付 between '2024-02-10' and '2024-02-14';

/*************************************************/
/*                  chaper 4                     */
/*************************************************/

select distinct 入金額 from 家計簿;
