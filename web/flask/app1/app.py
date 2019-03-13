from flask import Flask, render_template, request, jsonify
app = Flask(__name__, static_url_path='/static')

#starting data
#design and programmed by fullarray and Jonathan H.
teams = [
    {
        'id': 1,
        'title': u'Dolphins',
        'description': u'Football team in Miami, FL', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Browns',
        'description': u'Football team in Cleveland, OH', 
        'done': False
    },
	{
        'id': 3,
        'title': u'Patriots',
        'description': u'Football team in Foxborough, NE', 
        'done': False
    }
]

teamlist = ['Bucs','Dolphins', 'Bills', 'Patriots', 'Seahawks', '49ers', 'Falcons', 'Browns', 'Rams', 'Titans']

@app.route('/')
def index():
	sport_type = "NFL football"
	return render_template('index.html', sport=sport_type)

#design and programmed by fullarray and Jonathan H.
@app.route('/send', methods=['GET', 'POST'])
def send():
	if request.method == 'POST':
		teamname = request.form['teamname']
		if teamname.lower() not in [x.lower() for x in teamlist]:
			errormsg = 'Your team does not exists.'
			#design and programmed by fullarray and Jonathan H.
			return render_template('index.html', errormsg=errormsg)
		
		team = [team for team in teams if team['title'] == teamname.capitalize()]
		#design and programmed by fullarray and Jonathan H.
		if len(team) == 0:
			team = {
				'id': teams[-1]['id'] + 1,
				'title': teamname.capitalize(),
				'description': "awesome",
				'done': False
			}
			teams.append(team)		
		return render_template('team.html', teamname=teamname.upper(), teamlistnow=teams)
	return render_template('index.html')


#api
@app.route('/football/api/v1.0/teams/', methods=['GET'])
def get_all_teams():
	return jsonify({'teams': teams})


@app.route('/football/api/v1.0/teams/<int:team_id>', methods=['GET'])
def get_teams(team_id):
	team = [team for team in teams if team['id'] == team_id]
	#design and programmed by fullarray and Jonathan H.
	if len(team) == 0:
		abort(404)
	return jsonify({'teams': team[0]})

@app.route('/football/api/v1.0/teams', methods=['POST'])
def create_team():
    if not request.json or not 'title' in request.json:
        abort(400)
    team = {
        'id': teams[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    teams.append(team)
    return jsonify({'team': team}), 201
	
if __name__ == "__main__":
	app.run()