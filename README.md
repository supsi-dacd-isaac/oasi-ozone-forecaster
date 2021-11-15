# oasi-ozone-forecaster

## CURL examples to download forecasts results from ISAAC database:
<pre>
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_results WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case"
curl -G 'http://$HOST:$PORT/query?u=$USER&p=$PWD&pretty=true' --data-urlencode "db=$DB" --data-urlencode "q=SELECT * FROM o3_forecasts_probabilities WHERE flag_best='true' and time>='2021-05-13' and time<='2021-05-14' GROUP BY location, case, interval"
</pre>

## JSON sections and parameters**
