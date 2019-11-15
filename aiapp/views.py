from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def predict_rating(request):
    if request.method == 'POST':
        actor_1_facebook_likes = request.POST.get('actor_1_facebook_likes')
        print(actor_1_facebook_likes)
        return HttpResponse(actor_1_facebook_likes)
    else:
        return render(request, 'index.html', {})