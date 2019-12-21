from django.views.generic import TemplateView
#from django.shortcuts import render
#from django.http import HttpResponse

# Create your views here.
class HomePageView(TemplateView):
	#return HttpResponse('Hello, World!')
	template_name = 'home.html'

class AboutPageView(TemplateView):
	template_name = 'about.html'