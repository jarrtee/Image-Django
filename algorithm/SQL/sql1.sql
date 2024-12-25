select t.*,round(PieceNum*isnull(CONVERT(float,quo10),0),6) as money from
(
select *,ROW_NUMBER()over(partition by Cnname order by VestDate desc) num,case when CHARINDEX('/',ProductCode) != 0 then SUBSTRING(ProductCode,1,CHARINDEX('/',ProductCode)-1) else ProductCode end gg
from salary_count_formal
where f_type = 'auto' and PieceNum != 0 and Cnname is not null and VestDate >= dateadd(day,-2,getdate()) and pl05 = 'm1zp'
)t left join salary_quotas on quo09 = PriceCode
where num = 1 and round(PieceNum*isnull(CONVERT(float,quo10),0),6) <> 0