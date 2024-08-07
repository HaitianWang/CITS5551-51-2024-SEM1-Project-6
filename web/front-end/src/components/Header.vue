<template>
  <div id="Header">
    <div class="top-left-edition">
      <span class="user-info">
        <i class="el-icon-user" style="font-size: 23px"></i>
        <span id="name">{{ username }}</span>
      </span>
    </div>
    <nav class="middle">
      <router-link :to="{ path: '/uploadPage', query: { username: username } }" class="nav-link">Upload</router-link>
      <router-link :to="{ path: '/timeCapsule', query: { username: username } }" class="nav-link">TimeCapsule</router-link>
    </nav>
    <div id="word">
      <h1>{{ msg }}</h1>
      <i class="el-icon-s-home home-icon" v-on:click="goMainPage"></i>
    </div>
  </div>
</template>

<script>
  export default {
    name: "Header",
    props: ['username'],
    data() {
      return {
        msg: "Precision Agriculture: UWAIntelliCrop",
        activeIndex: "2",
      };
    },
    methods: {
      goFrontPage() {
        this.$router.push({
          path: "/frontPage",
          name: "frontPage",
        });
      },
      goMainPage() {
        this.$router.push({
          path: "/mainPage",
          name: "mainPage",
          query: {
            username: this.username
          }
        });
      },
    },
  };
</script>

<style scoped>
  #Header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0px 40px;
    width: 90%;
    margin: 10px auto;
    background-color: #6ec2b1; /* 绿色背景 */
    color: white; /* 白色文字 */
  }

  #word {
    display: flex;
    align-items: center;
    margin: 0;
  }

  h1 {
    color: white; /* 确保标题也是白色 */
    letter-spacing: 1px;
    font-size: 1.8em;
    margin-right: 10px; /* 确保图标与文字有间距 */
  }

  .el-menu-demo {
    width: 80%;
    margin: 0px auto;
    padding: 0px;
  }

  .top-left-edition .user-info {
    font-size: 20px;
    color: white;
    line-height: 24px;
    margin-right: 10px;
  }

  #name {
    margin-right: 3px;
  }

  .nav-link {
    margin: 0 15px;
    text-decoration: none; /* 去掉下划线 */
    font-size: 20px;
    color: white; /* 确保导航链接也是白色 */
  }

  .nav-link:hover {
    text-decoration: none; /* 保持悬停时无下划线 */
    color: #d9f0ea; /* 悬停时稍微变亮的颜色，可选 */
  }

  .home-icon {
    font-size: 30px; /* 调整图标大小 */
    cursor: pointer;
    color: white; /* 确保图标也是白色 */
  }

  .home-icon:hover {
    color: #d9f0ea; /* 悬停时稍微变亮的颜色，可选 */
  }
</style>
