-- This script lists all shows from hbtn_0d_tvshows_rate by their rating
select tv_shows.title, SUM(tv_show_ratings.rate) as rating from tv_shows INNER JOIN tv_show_ratings ON tv_show_ratings.show_id = tv_shows.id GROUP by tv_shows.title ORDER by rating DESC;
