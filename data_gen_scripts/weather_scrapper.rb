require "net/https"
require "uri"
require "json"
require "date"

# Getting weather information -------------------------------------------

out_file = File.new("weather.csv", "a")

(Date.new(2016, 12, 31)..Date.new(2017, 4, 1)).each do |date|
	dd = date.day.to_s
	dd = "0" + dd if date.day < 10
	mm = date.month.to_s
	mm = "0" + mm if date.month < 10
	yyyy = date.year.to_s
	date_str = "#{yyyy}#{mm}#{dd}"

	puts date

	uri = URI.parse("http://api.wunderground.com/api/7cd9d169a66ad891/history_" + date_str + "/q/NY/New_York.json")
	http = Net::HTTP.new(uri.host, uri.port)
	header = {'Accept' => 'application/json, text/javascript, */*; q=0.01' }
	request = Net::HTTP::Post.new(uri.request_uri, header)
	response = http.request(request)

	weather = JSON.parse(response.body)

	conds = ""
	temp = 0
	wind = 0
	humidity = 0
	visibility = 0
	weather["history"]["observations"].each do |observation|
		if (observation["conds"] != "Unknown")
			conds = observation["conds"]
		end
		if (observation["tempm"].to_i != -9999 && observation["tempm"] != "N/A")
			temp = observation["tempm"]
		end
		if (observation["wspdm"].to_f >= 0 && observation["wspdm"] != "N/A")
			wind = observation["wspdm"]
		end
		if (observation["hum"].to_f >= 0 && observation["hum"] != "N/A")
			humidity = observation["hum"]
		end
		if (observation["vism"].to_f >= 0 && observation["vism"] != "N/A")
			visibility = observation["vism"]
		end

		observation_date = "#{observation["date"]["year"]}-#{observation["date"]["mon"]}-#{observation["date"]["mday"]} #{observation["date"]["hour"]}:#{observation["date"]["min"]}:00"
		out_file.puts "#{observation_date},#{conds},#{temp},#{wind},#{humidity},#{visibility}"
	end

	sleep(6)


end

out_file.close