import Vue from 'vue'
import Router from 'vue-router'
import frontPage from "../components/frontPage";
import mainPage from "../components/mainPage";
import regiPage from "../components/regiPage";

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'frontPage',
      component: frontPage
    },
    {
      path: '/mainPage',
      name: 'mainPage',
      component: mainPage
    },
    {
      path: '/regiPage',
      name: 'regiPage',
      component: regiPage
    },
  ]
})
