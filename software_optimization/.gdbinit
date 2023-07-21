define pm 
	set $i = 0
	while $i < ($arg0).dim.y
		print ($arg0).data[$arg1][$i][0]@($arg0).dim.x
		set $i = $i + 1
	end
end

define pt
	set $i = 0
	while $i < ($arg0).dim.z
		set $j = 0
		while $j < ($arg0).dim.y
			print ($arg0).data[$i][$j][0]@($arg0.dim.x)
			set $j = $j +1
		end
		set $i = $i + 1
	end
end

define pmc
	set $i = 0
	while $i < ($arg0).size[1]
		print ($arg0).data[$arg1][$i][0]@($arg0).size[2]
		set $i = $i + 1
	end
end


define ptc
	set $i = 0
	while $i < ($arg0).size[0]
		set $j = 0
		while $j < ($arg0).size[1]
			print ($arg0).data[$i][$j][0]@(($arg0).size[2])
			set $j = $j + 1
		end
		set $i = $i + 1
	end
end
