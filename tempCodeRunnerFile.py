# get form data
    year = int(request.form.get('year'))
    engine_size = float(request.form.get('engine_size'))
    cylinders = int(request.form.get('cylinders'))
    transmission = request.form.get('transmission')
    fuel = request.form.get('fuel')
    coemissions = float(request.form.get('coemissions'))
    make = request.form.get('make')
    model_name = request.form.get('model')
    vehicle_class = request.form.get('vehicle_class')